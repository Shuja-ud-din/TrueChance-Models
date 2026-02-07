import runpod
import base64
import tempfile
from runpod.serverless.utils import rp_cuda
from TTS.api import TTS
import torch
import os

print("Initializing Coqui TTS worker...")

# -------------------------
# Device selection
# -------------------------
DEVICE = "cuda" if rp_cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# -------------------------
# Constants
# -------------------------
XTTS_MODEL_PATH = "/app/models/xtts_v2"
XTTS_CONFIG_PATH = "/app/models/xtts_v2/config.json"
EN_SPEAKER = "Andrew Chipper"

# -------------------------
# Model registry
# -------------------------
tts_engines = {}

def load_models():
    """
    Load all TTS models at startup.
    Avoids cold-start latency.
    """

    # English – XTTS v2 (Andrew Chipper)
    print("Loading English XTTS v2 (Andrew Chipper)...")
    tts_engines["en"] = TTS(
        model_path=XTTS_MODEL_PATH,
        config_path=XTTS_CONFIG_PATH
    ).to(DEVICE)

    # Arabic – Tacotron2
    print("Loading Arabic model -> tts_models/ar/mai/tacotron2")
    tts_engines["ar"] = TTS(
        model_name="tts_models/ar/mai/tacotron2"
    ).to(DEVICE)

    print("All models loaded successfully.")

# Load once at container start
load_models()

# -------------------------
# TTS synthesis
# -------------------------
def synthesize(text: str, lang: str):
    if lang not in tts_engines:
        raise ValueError(f"Unsupported language: {lang}")

    tts = tts_engines[lang]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name

    if lang == "en":
        # XTTS v2 – Andrew Chipper
        tts.tts_to_file(
            text=text,
            speaker=EN_SPEAKER,
            language="en",
            file_path=out_path,
            temperature=0.55,
            top_p=0.85,
            top_k=50,
            speed=1.0,
        )
    else:
        # Arabic Tacotron2
        tts.tts_to_file(
            text=text,
            file_path=out_path
        )

    with open(out_path, "rb") as f:
        audio_bytes = f.read()

    return base64.b64encode(audio_bytes).decode("utf-8")

# -------------------------
# RunPod handler
# -------------------------
def handler(event):
    """
    Expected input:
    {
      "input": {
        "text": "Hello world",
        "lang": "en" | "ar"
      }
    }
    """

    input_data = event.get("input", {})
    text = input_data.get("text")
    lang = input_data.get("lang", "en")

    if not text:
        return {"error": "text is required"}

    try:
        audio_base64 = synthesize(text, lang)
    except Exception as e:
        return {"error": str(e)}

    return {
        "audio": audio_base64,
        "format": "wav",
        "lang": lang
    }

# -------------------------
# Concurrency control
# -------------------------
def adjust_concurrency(current_concurrency):
    return 6 if DEVICE == "cuda" else 1

# -------------------------
# RunPod start
# -------------------------
if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler,
        "adjust_concurrency": adjust_concurrency
    })
