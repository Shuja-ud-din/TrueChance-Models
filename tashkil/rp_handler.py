import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

MODEL_NAME = "glonor/byt5-arabic-diacritization"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

print("Model loaded successfully.")

class DiacritizeRequest(BaseModel):
    text: str

class DiacritizeResponse(BaseModel):
    text: str

# ✅ Required health check for RunPod
@app.get("/ping")
async def ping():
    return {"status": "healthy"}

@app.post("/diacritize", response_model=DiacritizeResponse)
async def diacritize(req: DiacritizeRequest):
    inputs = tokenizer(req.text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"text": result}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 80))
    uvicorn.run(app, host="0.0.0.0", port=port)
