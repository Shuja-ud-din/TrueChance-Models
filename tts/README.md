# Coqui TTS Worker for RunPod

A serverless Text-to-Speech worker for [RunPod](https://runpod.io) using Coqui's XTTS v2 model. This worker provides high-quality, multi-language speech synthesis with GPU acceleration support.

## Features

- XTTS v2 model for natural-sounding speech
- Multi-language support (English, Arabic, and more)
- Multiple speaker voices
- CUDA GPU acceleration (with CPU fallback)
- Base64 encoded audio output
- RunPod serverless deployment ready

## Project Structure

```
Coqui-TTS-Worker/
├── Dockerfile          # Docker container configuration
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── rp_handler.py       # Main RunPod worker handler
├── test_input.json     # Sample test input
├── .gitignore          # Git ignore rules
└── models/             # Model files (not in git)
    └── xtts_v2/        # XTTS v2 model directory
        ├── config.json
        ├── model.pth
        ├── vocab.json
        └── ...
```

## Prerequisites

- Docker installed on your system
- Python 3.10+ (for local development)
- Hugging Face account (for model download)
- RunPod account (for deployment)

---

## Step 1: Download the XTTS v2 Model

> **IMPORTANT:** The automated model download script will NOT work due to Coqui's Terms of Service (TOS) agreement requirement. You must manually download the model first.

### 1.1 Accept the TOS on Hugging Face

1. Go to [https://huggingface.co/coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2)
2. Log in to your Hugging Face account
3. Accept the Terms of Service agreement on the model page

### 1.2 Download the Model

**Option A: Using Hugging Face CLI (Recommended)**

```bash
# Install huggingface-cli if not already installed
pip install huggingface_hub

# Login to Hugging Face
huggingface-cli login

# Download the model
huggingface-cli download coqui/XTTS-v2 --local-dir ./models/xtts_v2
```

**Option B: Using Python Script**

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="coqui/XTTS-v2",
    local_dir="./models/xtts_v2",
    local_dir_use_symlinks=False
)
```

**Option C: Using TTS Library**

```bash
# First, install TTS
pip install TTS==0.22.0

# Download model (this will prompt for TOS acceptance)
tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 --list_speaker_idxs
```

After running this command, the model will be downloaded to your cache directory (usually `~/.local/share/tts/` on Linux or `C:\Users\<username>\AppData\Local\tts\` on Windows). Copy the contents to `./models/xtts_v2/`.

### 1.3 Verify Model Files

After downloading, ensure your `models/xtts_v2/` directory contains these files:

```
models/xtts_v2/
├── config.json
├── model.pth
├── vocab.json
├── speakers_xtts.pth
├── hash.md5
└── ...
```

---

## Step 2: Local Development Setup (Optional)

If you want to test locally before building the Docker image:

### 2.1 Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate
```

### 2.2 Install Dependencies

```bash
pip install -r requirements.txt
```

### 2.3 Install System Dependencies

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg espeak-ng libsndfile1
```

**On Windows:**
- Install FFmpeg: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
- Install espeak-ng: Download from [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases)

**On macOS:**
```bash
brew install ffmpeg espeak-ng libsndfile
```

### 2.4 Run Locally

```bash
python rp_handler.py
```

---

## Step 3: Build the Docker Image

### 3.1 Build the Image

```bash
# Build the Docker image
docker build -t coqui-tts-worker .
```

### 3.2 Build with Custom Tag

```bash
# Build with version tag
docker build -t coqui-tts-worker:v1.0.0 .

# Build with your Docker Hub username
docker build -t yourusername/coqui-tts-worker:latest .
```

### 3.3 Verify the Build

```bash
# List images to verify
docker images | grep coqui-tts-worker
```

---

## Step 4: Test the Docker Image Locally

### 4.1 Run the Container

```bash
# Run with GPU support (NVIDIA)
docker run --gpus all -p 8000:8000 coqui-tts-worker

# Run without GPU (CPU only)
docker run -p 8000:8000 coqui-tts-worker
```

### 4.2 Test with Sample Input

```bash
# Using curl
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "Hello, this is a test.", "language": "en"}}'
```

---

## Step 5: Push to Docker Registry

### 5.1 Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Tag the image
docker tag coqui-tts-worker yourusername/coqui-tts-worker:latest

# Push to Docker Hub
docker push yourusername/coqui-tts-worker:latest
```

### 5.2 Push to RunPod Registry (Alternative)

Follow RunPod's documentation for pushing to their container registry.

---

## Step 6: Deploy to RunPod

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Create a new Serverless Endpoint
3. Select your Docker image: `yourusername/coqui-tts-worker:latest`
4. Configure GPU type (recommended: RTX 3090, RTX 4090, or A100)
5. Set the following environment variables if needed:
   - `RUNPOD_DEBUG_LEVEL`: `DEBUG` (optional, for debugging)
6. Deploy the endpoint

---

## API Usage

### Request Format

```json
{
  "input": {
    "text": "Your text to synthesize",
    "language": "en",
    "speaker": "Andrew Chipper"
  }
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | Yes | - | The text to convert to speech |
| `language` | string | No | `"en"` | Language code (e.g., `"en"`, `"ar"`) |
| `speaker` | string | No | Auto | Speaker name (uses default for language if not specified) |

### Default Speakers

| Language | Default Speaker |
|----------|-----------------|
| `en` (English) | Andrew Chipper |
| `ar` (Arabic) | Badr Odhiambo |
| Other | Gracie Wise |

### Response Format

```json
{
  "audio": "UklGRi4AAABXQVZFZm10IBAAAAABAAEA...",
  "format": "wav",
  "language": "en",
  "speaker": "Andrew Chipper"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `audio` | string | Base64-encoded WAV audio data |
| `format` | string | Audio format (always `"wav"`) |
| `language` | string | Language used for synthesis |
| `speaker` | string | Speaker name used |

### Error Response

```json
{
  "error": "text is required"
}
```

---

## Example: Decoding Audio Response

### Python

```python
import base64
import requests

# Make request to RunPod endpoint
response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={"input": {"text": "Hello world!", "language": "en"}}
)

# Decode and save audio
audio_base64 = response.json()["output"]["audio"]
audio_bytes = base64.b64decode(audio_base64)

with open("output.wav", "wb") as f:
    f.write(audio_bytes)
```

### JavaScript/Node.js

```javascript
const response = await fetch('https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    input: { text: 'Hello world!', language: 'en' }
  })
});

const data = await response.json();
const audioBuffer = Buffer.from(data.output.audio, 'base64');
require('fs').writeFileSync('output.wav', audioBuffer);
```

---

## Supported Languages

XTTS v2 supports the following languages:

- English (`en`)
- Spanish (`es`)
- French (`fr`)
- German (`de`)
- Italian (`it`)
- Portuguese (`pt`)
- Polish (`pl`)
- Turkish (`tr`)
- Russian (`ru`)
- Dutch (`nl`)
- Czech (`cs`)
- Arabic (`ar`)
- Chinese (`zh-cn`)
- Japanese (`ja`)
- Hungarian (`hu`)
- Korean (`ko`)

---

## Troubleshooting

### Model Download Issues

**Error: "You need to agree to the terms of service"**
- Visit [Hugging Face XTTS-v2](https://huggingface.co/coqui/XTTS-v2) and accept the TOS
- Make sure you're logged in with `huggingface-cli login`

**Error: "Model files not found"**
- Verify the `models/xtts_v2/` directory exists and contains all required files
- Check that `config.json` and `model.pth` are present

### Docker Build Issues

**Error: "COPY failed: file not found"**
- Ensure the `models/` directory exists with the XTTS v2 model
- Run the model download steps before building

### Runtime Issues

**Error: "CUDA out of memory"**
- Use a GPU with more VRAM (recommended: 8GB+)
- Reduce concurrency in `adjust_concurrency()` function

**Error: "espeak-ng not found"**
- Ensure system dependencies are installed
- For Docker, verify the Dockerfile includes the apt-get install commands

---

## Configuration

### Concurrency Settings

Edit `rp_handler.py` to modify concurrency:

```python
def adjust_concurrency(current_concurrency):
    return 2  # Change this value
```

### Adding Custom Speakers

Modify `DEFAULT_SPEAKERS` in `rp_handler.py`:

```python
DEFAULT_SPEAKERS = {
    "en": "Andrew Chipper",
    "ar": "Badr Odhiambo",
    "es": "Your Spanish Speaker",
    # Add more languages...
}
```

---

## License

This project uses Coqui's XTTS v2 model which is subject to Coqui's license terms. Please review and comply with their license agreement.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions:
- Open an issue on GitHub
- Check RunPod documentation for deployment issues
- Review Coqui TTS documentation for model-related questions
