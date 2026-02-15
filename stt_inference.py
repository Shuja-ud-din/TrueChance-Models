import modal
from typing import Optional
from pydantic import BaseModel

app = modal.App("whisper-inference")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .pip_install(
        "fastapi[standard]==0.95.1",
        "uvicorn",
        "faster-whisper",
    )
)

class STTRequest(BaseModel):
    audio_base64: str
    language: Optional[str] = None


@app.cls(
    image=image,
    gpu="L40S",
    min_containers=0,
    max_containers=1,
    scaledown_window=30,
)
@modal.concurrent(max_inputs=3)
class WhisperModel:

    @modal.enter()
    def load_model(self):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            "medium",
            device="cuda",
            compute_type="float16",
        )

    @modal.fastapi_endpoint(method="POST", docs=True, requires_proxy_auth=True)
    async def transcribe(self, req: STTRequest):    
        import base64
        import numpy as np

        language = req.language if req.language else "en"

        audio_bytes = base64.b64decode(req.audio_base64)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        segments, _ = self.model.transcribe(
            audio,
            beam_size=2,
            language=language,
            condition_on_previous_text=False,
            vad_filter=False,
        )

        text = "".join(seg.text for seg in segments)

        return {"text": text, "language": language}