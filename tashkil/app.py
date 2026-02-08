import os
import asyncio
import time
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ================= CONFIG =================
MODEL_NAME = "glonor/byt5-arabic-diacritization"

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))          # micro-batch size
BATCH_WAIT_MS = int(os.getenv("BATCH_WAIT_MS", 8))    # wait time to collect batch
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 1024))

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= APP =================
app = FastAPI()

print("🚀 Loading ByT5 model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Enable faster inference
torch.backends.cudnn.benchmark = True

print("✅ Model loaded on", device)

# ================= REQUEST QUEUE =================
request_queue = asyncio.Queue()

class DiacritizeRequest(BaseModel):
    text: str

class DiacritizeResponse(BaseModel):
    text: str
    latency_ms: float

# ================= HEALTH CHECK =================
@app.get("/ping")
async def ping():
    return {"status": "healthy", "device": device}

# ================= WORKER LOOP =================
async def batch_worker():
    while True:
        batch = []
        futures = []

        start_time = time.time()

        # Collect first request
        req, fut = await request_queue.get()
        batch.append(req)
        futures.append(fut)

        # Collect more requests within time window
        while len(batch) < BATCH_SIZE:
            try:
                timeout = BATCH_WAIT_MS / 1000 - (time.time() - start_time)
                if timeout <= 0:
                    break
                req, fut = await asyncio.wait_for(request_queue.get(), timeout)
                batch.append(req)
                futures.append(fut)
            except asyncio.TimeoutError:
                break

        texts = [r.text for r in batch]

        # Tokenization (CPU)
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        # GPU inference
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                num_beams=4,
                 length_penalty=0.6,
                early_stopping=False, 
                no_repeat_ngram_size=3,
            )

        results = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Return results
        for fut, text in zip(futures, results):
            fut.set_result(text)

# Start background batch worker
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_worker())

# ================= API ENDPOINT =================
@app.post("/diacritize", response_model=DiacritizeResponse)
async def diacritize(req: DiacritizeRequest):
    loop = asyncio.get_event_loop()
    fut = loop.create_future()

    start = time.time()
    await request_queue.put((req, fut))
    result = await fut
    latency = (time.time() - start) * 1000

    return {"text": result, "latency_ms": round(latency, 2)}

# ================= MAIN =================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 80))
    uvicorn.run(app, host="0.0.0.0", port=port)
