from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import time

from services.tts_service import synthesize
from services.tashkeel_service import diacritize

router = APIRouter()

class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    speaker: str | None = None


@router.post("/tts")
async def tts_endpoint(req: TTSRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="text is required")

    total_start = time.time()

    text = req.text
    tashkeel_latency = 0

    # 🔥 Apply tashkeel only for Arabic
    if req.language == "ar":
        text, tashkeel_latency = await diacritize(text)

    try:
        audio, speaker, tts_latency = await synthesize(
            text,
            req.language,
            req.speaker
        )
    except Exception as e:
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
