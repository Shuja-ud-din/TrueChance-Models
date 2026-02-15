import os

# Server ports (RunPod)
PORT = int(os.getenv("PORT", "80"))
PORT_HEALTH = int(os.getenv("PORT_HEALTH", str(PORT)))

# GPU concurrency
MAX_GPU_CONCURRENCY = int(os.getenv("MAX_GPU_CONCURRENCY", "4"))

# XTTS paths
XTTS_MODEL_DIR = os.getenv("XTTS_MODEL_DIR", "models/xtts_v2")
XTTS_CONFIG_PATH = f"{XTTS_MODEL_DIR}/config.json"

# Default speakers
DEFAULT_SPEAKERS = {
    "en": "Andrew Chipper",
    "ar": "Badr Odhiambo"
}
