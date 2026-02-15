from fastapi import FastAPI
import uvicorn

from config import PORT
from api.routes import router
from services.tts_service import load_tts
from services.tashkeel_service import load_tashkeel

app = FastAPI()

# Register routes
app.include_router(router)

# Health endpoint required by RunPod
@app.get("/ping")
async def ping():
    return {"status": "healthy"}

# Load model at startup
load_tts()
load_tashkeel()

if __name__ == "__main__":
    print(f"🌐 Starting XTTS Load Balancer on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
