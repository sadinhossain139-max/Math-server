import os
import re
import logging
import time
from typing import Optional, Dict, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# ---------------------------
# Configuration & Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai-server")

# Environment variables (set these in your deployment)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "500"))

# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(
    title="AI Calculation Server",
    description="Lightweight stateless AI API for math problem solving",
    version="1.0.0",
)

# CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your app's domain
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ---------------------------
# Request Validation
# ---------------------------
class CalculateRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)
    mode: str = Field(..., pattern="^(detailed|answer_only|roadmap)$")

    @validator("question")
    def sanitize_question(cls, v: str) -> str:
        # Remove control characters and limit to safe set
        v = re.sub(r"[\x00-\x1f\x7f]", "", v).strip()
        # Allow Bengali, English, numbers, common math symbols, basic punctuation
        allowed = re.compile(r'[^a-zA-Z0-9\u0980-\u09FF\s\.\,\+\-\*\/\(\)\[\]\{\}\^\=\%\<\>\?\!\:\;\|\\\']')
        v = allowed.sub("", v)
        if not v:
            raise ValueError("প্রশ্ন খালি বা অবৈধ অক্ষর রয়েছে")
        if len(v) > MAX_QUESTION_LENGTH:
            raise ValueError(f"প্রশ্ন খুব দীর্ঘ (সর্বোচ্চ {MAX_QUESTION_LENGTH} অক্ষর)")
        return v

# ---------------------------
# Rate Limiter (in-memory, per IP)
# ---------------------------
class RateLimiter:
    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests: Dict[str, Tuple[float, int]] = {}  # ip -> (window_start, count)

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        window_start, count = self.requests.get(ip, (now, 0))
        if now - window_start > self.window_seconds:
            # Reset window
            self.requests[ip] = (now, 1)
            return True
        if count >= self.limit:
            return False
        self.requests[ip] = (window_start, count + 1)
        return True

rate_limiter = RateLimiter(limit=RATE_LIMIT_PER_MINUTE)

async def check_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(status_code=429, detail="অনেক বেশি রিকোয়েস্ট, কিছুক্ষণ পরে আবার চেষ্টা করুন")

# ---------------------------
# System Prompts
# ---------------------------
SYSTEM_PROMPTS = {
    "detailed": "তুমি একজন অভিজ্ঞ গণিত শিক্ষক। নিচের প্রশ্নের ধাপে ধাপে সমাধান দাও, প্রতিটি সূত্র ব্যাখ্যা করো।",
    "answer_only": "শুধুমাত্র চূড়ান্ত উত্তরটি দাও, কোনো ব্যাখ্যা ছাড়া।",
    "roadmap": "উত্তর দিয়ো না, শুধু কোন কোন সূত্র ও ধাপ অনুসরণ করতে হবে তা সংক্ষেপে বলো।",
}

def get_system_prompt(mode: str) -> str:
    return SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["detailed"])

# ---------------------------
# AI Provider Calls (async)
# ---------------------------
async def call_groq(question: str, system_prompt: str) -> str:
    """Call Groq API, return answer text."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

async def call_gemini(question: str, system_prompt: str) -> str:
    """Call Gemini API, return answer text."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": system_prompt},
                    {"text": question},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024,
        },
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        # Extract text from first candidate
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

async def get_ai_answer(question: str, mode: str) -> str:
    """Try Groq first, then Gemini; raise if both fail."""
    system_prompt = get_system_prompt(mode)
    errors = []

    # Try Groq if key exists
    if GROQ_API_KEY:
        try:
            logger.info(f"Calling Groq with mode={mode}")
            return await call_groq(question, system_prompt)
        except Exception as e:
            logger.error(f"Groq failed: {e}")
            errors.append(f"Groq: {str(e)}")
    else:
        errors.append("Groq API key missing")

    # Try Gemini if key exists
    if GEMINI_API_KEY:
        try:
            logger.info(f"Calling Gemini with mode={mode}")
            return await call_gemini(question, system_prompt)
        except Exception as e:
            logger.error(f"Gemini failed: {e}")
            errors.append(f"Gemini: {str(e)}")
    else:
        errors.append("Gemini API key missing")

    raise RuntimeError("All AI providers failed: " + "; ".join(errors))

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/")
async def health_check():
    return {"status": "ok", "service": "AI Calculation Server"}

@app.post("/api/calculate", dependencies=[Depends(check_rate_limit)])
async def calculate(request: CalculateRequest):
    """
    Process a math question with the specified mode and return the AI answer.
    """
    try:
        answer = await get_ai_answer(request.question, request.mode)
        return {"answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="সার্ভারে ত্রুটি হয়েছে, পরে আবার চেষ্টা করুন")

# ---------------------------
# Run (for local development)
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)