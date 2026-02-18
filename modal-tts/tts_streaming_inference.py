import modal
from pydantic import BaseModel

volume = modal.Volume.from_name("xtts-model")

image = modal.Image.from_registry(
        "nvidia/cuda:12.2.0-runtime-ubuntu22.04",
        add_python="3.10"
    ).apt_install("git", "ffmpeg").pip_install(
            "fastapi==0.110.0",
            "uvicorn==0.29.0",
            "pydantic==2.6.4",
            "TTS==0.22.0",
            "soundfile",
            "transformers==4.36.2",
            "torch==2.1.2",
            "torchaudio==2.1.2",
            "camel-tools",
        )


image = image.add_local_dir("build", remote_path="/root/build", copy=True)
image = image.run_commands(
    "python /root/build/download_camel.py"
)

app = modal.App("tts-streaming-inference")

class SynthesizeRequest(BaseModel):
    text: str
    language: str = "en"

TTS_WARMUP_TEXT = "This is a warm-up sentence to load the TTS model."
TASHKEEL_WARMUP_TEXT = "مرحبا بك"

DEFAULT_SPEAKERS = {
    "en": "Andrew Chipper",
    "ar": "Badr Odhiambo"
}


with image.imports():
    import time
    import asyncio
    import torch
    import io
    import soundfile as sf
    import base64

    from fastapi import FastAPI, HTTPException

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
    from camel_tools.tagger.default import DefaultTagger
    from camel_tools.tokenizers.word import simple_word_tokenize


@app.cls(
    image=image,
    gpu="L40S",
    min_containers=0,
    max_containers=3,
    scaledown_window=60,
    volumes={"/model": volume},
)
@modal.concurrent(max_inputs=3)
class CoquiTTS:
    @modal.enter()
    async def load(self):
        print("Loading XTTS v2...")

        config = XttsConfig()
        config.load_json("/model/xtts_v2/config.json")
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir="/model/xtts_v2")
        self.model.cuda()
        self.model.eval()

        self.speaker_latents = {}

        for lang, speaker_name in DEFAULT_SPEAKERS.items():
            print(f"Precomputing speaker: {speaker_name}")

            # get conditioning latents using speaker_id
            latents = self.model.speaker_manager.speakers[speaker_name]

            gpt_latent = latents['gpt_cond_latent'].cuda()
            speaker_embedding = latents['speaker_embedding'].cuda()

            self.speaker_latents[lang] = {
                "gpt": gpt_latent,
                "speaker": speaker_embedding,
            }


        # self.tts_model = TTS(
        #     model_path="/model/xtts_v2",
        #     config_path="/model/xtts_v2/config.json",
        #     progress_bar=False,
        #     gpu=True
        # ).to("cuda")

        print("Loading Tashkeel model...")

        self.disambiguator = BERTUnfactoredDisambiguator.pretrained(
            model_name='msa',
            use_gpu=True
        )
        self.tagger = DefaultTagger(self.disambiguator, 'diac')

        print("Warming up TTS model...")
        await self.text_to_speech(TTS_WARMUP_TEXT, language="en")

        print("Warming up Tashkeel model...")
        await self.diacritize(TASHKEEL_WARMUP_TEXT)

        print("✅ Models warmed up and ready for inference.")    

    
    @modal.method()
    async def add_tashkeel(self, text: str):
        output = await self.diacritize(text)
        
        return output

    async def diacritize(self, text: str):
        tokens = simple_word_tokenize(text)

        start = time.time()
        diacritized_tokens = await asyncio.to_thread(
            self.tagger.tag,
            tokens,
        )

        result = " ".join(diacritized_tokens)
        latency = (time.time() - start) * 1000

        return result, latency
    
    async def text_to_speech(self, text: str, language: str):
        try:
            start_time = time.time()
            latents = self.speaker_latents.get(language)

            if not latents:
                raise ValueError(f"Unsupported language: {language}")

            with torch.inference_mode():
                output = self.model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=latents["gpt"],
                    speaker_embedding=latents["speaker"],
                    temperature=0.7,
                )

            audio = output["wav"]

            buf = io.BytesIO()
            sf.write(buf, audio, samplerate=24000, format='WAV')
            audio_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            latency = (time.time() - start_time) * 1000

            return audio_base64, latency
        except Exception as e:
            print(f"Error during TTS synthesis: {e}")
            raise HTTPException(status_code=500, detail="TTS synthesis failed")

    @modal.asgi_app()
    def web(self):

        web_app = FastAPI()

        @web_app.get("/ping")
        async def ping():
            return {"status": "ok", "message": "TTS API is alive!"}
        
        @web_app.post("/synthesize")
        async def synthesize(req: SynthesizeRequest):
            if not req.text:
                raise HTTPException(status_code=400, detail="text is required")

            total_start = time.time()

            text = req.text
            tashkeel_latency = 0

            # 🔥 Apply tashkeel only for Arabic
            if req.language == "ar":
                text, tashkeel_latency = await self.diacritize(text)

            try:
                audio, tts_latency = await self.text_to_speech(
                    text,
                    req.language,
                )

            except Exception as e:
                print(f"Error in synthesis: {e}")
                raise HTTPException(status_code=500, detail=str(e))

            total_latency = (time.time() - total_start) * 1000

            return {
                "audio": audio,
                "tashkeel_latency": tashkeel_latency,
                "tts_latency": tts_latency,
                "total_latency": total_latency
            }
        
        return web_app