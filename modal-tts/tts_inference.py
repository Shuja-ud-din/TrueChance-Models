from pathlib import Path
import modal
from pydantic import BaseModel
import asyncio

volume = modal.Volume.from_name("xtts-model")

image = modal.Image.from_registry(
        "nvidia/cuda:12.2.0-runtime-ubuntu22.04",
        add_python="3.10",
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


app = modal.App("tts-inference")

class SynthesizeRequest(BaseModel):
    text: str
    language: str = "en"

# Default speakers
DEFAULT_SPEAKERS = {
    "en": "Andrew Chipper",
    "ar": "Badr Odhiambo"
}

gpu_semaphore = asyncio.Semaphore(8)
tashkeel_semaphore = asyncio.Semaphore(2)

with image.imports():
    import time
    import io
    import base64
    import tempfile

    import soundfile as sf
    from TTS.api import TTS
    from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
    from camel_tools.tagger.default import DefaultTagger
    from camel_tools.tokenizers.word import simple_word_tokenize
    

@app.cls(
    image=image,
    gpu="A100-80GB",
    min_containers=0,
    max_containers=3,
    scaledown_window=300,
    volumes={"/model": volume},
)
@modal.concurrent(max_inputs=10)
class CoaquiTTS:
    @modal.enter()
    def load(self):
        print("Loading XTTS v2...")

        self.tts_model = TTS(
            model_path="/model/xtts_v2",
            config_path="/model/xtts_v2/config.json",
            progress_bar=False
        ).to("cuda")

        print("Loading Tashkeel model...")

        self.disambiguator = BERTUnfactoredDisambiguator.pretrained(
            model_name='msa',
            use_gpu=True
        )
        self.tagger = DefaultTagger(self.disambiguator, 'diac')

        print("✅ Tashkeel model loaded.")

    
    @modal.method()
    async def add_tashkeel(self, text: str):
        output = await self.diacritize(text)
        
        return output

    async def diacritize(self, text: str):
        async with tashkeel_semaphore:
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
            speaker = DEFAULT_SPEAKERS.get(language)

            start_time = time.time()

            async with gpu_semaphore:

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    out_path = f.name

                # Run blocking TTS in thread pool
                await asyncio.to_thread(
                    self.tts_model.tts_to_file,
                    file_path=out_path,
                    text=text,
                    speaker=speaker,
                    language=language,
                )

                with open(out_path, "rb") as f:
                    audio_bytes = f.read()

                audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

            latency_ms = (time.time() - start_time) * 1000

            return {
                "audio": audio_base64,
                "format": "wav",
                "language": language,
                "speaker": speaker,
                "latency_ms": round(latency_ms, 2),
            }
        
        except Exception as e:
            print(f"Error in text_to_speech: {e}")
            raise


    @modal.asgi_app(requires_proxy_auth=True)
    def web(self):
        from fastapi import FastAPI, HTTPException

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
                audio_result = await self.text_to_speech(
                    text,
                    req.language,
                )

                audio = audio_result["audio"]
                speaker = audio_result["speaker"]
                tts_latency = audio_result["latency_ms"]

            except Exception as e:
                print(f"Error in synthesis: {e}")
                raise HTTPException(status_code=500, detail=str(e))

            total_latency = (time.time() - total_start) * 1000

            return {
                "audio": audio,
                "format": "wav",
                "language": req.language,
                "speaker": speaker,
                "text": text,
                "latency": {
                    "tashkeel_ms": round(tashkeel_latency, 2),
                    "tts_ms": round(tts_latency, 2),
                    "total_ms": round(total_latency, 2)
                }
            }
        
        return web_app
