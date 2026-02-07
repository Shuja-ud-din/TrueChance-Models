import os
import base64
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from faster_whisper import WhisperModel
from runpod.serverless.utils import rp_cuda

app = FastAPI()

# -----------------------------
# Request / Response Models
# -----------------------------
class TranscriptionRequest(BaseModel):
    audio_base64: str
    language: str | None = None

class TranscriptionResponse(BaseModel):
    text: str
    language: str

# -----------------------------
# Model Load (ONCE per worker)
# -----------------------------
print("Starting Whisper worker...")

DEVICE = "cuda" if rp_cuda.is_available() else "cpu"

model = WhisperModel(
    "medium",
    device=DEVICE,
    compute_type="float16" if DEVICE == "cuda" else "int8"
)

print("Whisper medium model loaded on", DEVICE)

# -----------------------------
# Health Check
# -----------------------------
@app.get("/ping")
async def ping():
    return {"status": "healthy"}

# -----------------------------
# Transcription Endpoint
# -----------------------------
@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(req: TranscriptionRequest):
    try:
        # Decode base64 → raw PCM16
        audio_bytes = base64.b64decode(req.audio_base64)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        segments, info = model.transcribe(
            audio,
            beam_size=2,
            language=req.language
        )

        text = "".join(segment.text for segment in segments)

        return {
            "text": text,
            "language": info.language
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 80))
    uvicorn.run(app, host="0.0.0.0", port=port)