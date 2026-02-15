import asyncio
import tempfile
import base64
import time
from TTS.api import TTS
from core.device import DEVICE
from config import MAX_GPU_CONCURRENCY, DEFAULT_SPEAKERS

gpu_semaphore = asyncio.Semaphore(MAX_GPU_CONCURRENCY)

tts_model = None


def load_tts():
    global tts_model
    print("🎙 Loading XTTS v2...")
    tts_model = TTS(
        model_path="models/xtts_v2",
        config_path="models/xtts_v2/config.json",
        progress_bar=False
    ).to(DEVICE)
    print("✅ XTTS ready.")


async def synthesize(text: str, language: str, speaker: str | None):
    if speaker is None:
        speaker = DEFAULT_SPEAKERS.get(language, "Gracie Wise")

    async with gpu_semaphore:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name

        start = time.time()

        await asyncio.to_thread(
            tts_model.tts_to_file,
            text=text,
            file_path=out_path,
            speaker=speaker,
            language=language
        )

        latency = (time.time() - start) * 1000

        with open(out_path, "rb") as f:
            audio_bytes = f.read()

    return base64.b64encode(audio_bytes).decode("utf-8"), speaker, latency
