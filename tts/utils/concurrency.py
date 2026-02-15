import asyncio
from config import MAX_GPU_CONCURRENCY

gpu_semaphore = asyncio.Semaphore(MAX_GPU_CONCURRENCY)
