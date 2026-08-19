import os
import re
import sys
import logging
import time
import json
from typing import Optional, Dict, Tuple, List, Any
from datetime import datetime, timedelta
from collections import defaultdict

# Fix for Python 3.14 event loop policy
if sys.version_info >= (3, 14):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import httpx
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, ConfigDict

# ---------------------------
# Configuration & Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ai_server.log")
    ]
)
logger = logging.getLogger("ai-server")

# Environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "500"))
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

# Free fallback provider
FREE_MODEL_ENABLED = os.getenv("FREE_MODEL_ENABLED", "true").lower() == "true"
FREE_MODEL_NAME = os.getenv("FREE_MODEL_NAME", "openai")

# Groq model list for fallback
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(
    title="AI Calculation Server - Enhanced",
    description="Advanced stateless AI API for mathematical problem solving",
    version="3.0.1",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ---------------------------
# Request Validation
# ---------------------------
class CalculateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)
    mode: str = Field(..., pattern="^(detailed|answer_only|roadmap|interactive|quiz)$")
    language: str = Field("bn", pattern="^(bn|en)$")
    difficulty: str = Field("auto", pattern="^(auto|basic|intermediate|advanced)$")

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        v = re.sub(r"[\x00-\x1f\x7f]", "", v).strip()
        allowed = re.compile(r'[^a-zA-Z0-9\u0980-\u09FF\s\.\,\+\-\*\/\(\)\[\]\{\}\^\=\%\<\>\?\!\:\;\|\\\'\"\@\#\$\&\_\~\`]')
        v = allowed.sub("", v)
        if not v:
            raise ValueError("Question is empty")
        return v

class BatchCalculateRequest(BaseModel):
    questions: List[CalculateRequest] = Field(..., min_length=1, max_length=10)
    parallel: bool = Field(True)

# ---------------------------
# Rate Limiter
# ---------------------------
class RateLimiter:
    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        self.requests[ip] = [t for t in self.requests[ip] if now - t < self.window_seconds]
        if len(self.requests[ip]) >= self.limit:
            return False
        self.requests[ip].append(now)
        return True

rate_limiter = RateLimiter(limit=RATE_LIMIT_PER_MINUTE)

async def check_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests")

# ---------------------------
# System Prompts (English to avoid encoding issues)
# ---------------------------
SYSTEM_PROMPTS = {
    "detailed": {
        "bn": "You are an experienced math teacher. Solve step by step in Bengali. Explain each formula clearly. Show final answer.",
        "en": "You are an experienced math teacher. Solve step by step with explanations."
    },
    "answer_only": {
        "bn": "Give only the final answer in Bengali, no explanation.",
        "en": "Give only the final answer, no explanation."
    },
    "roadmap": {
        "bn": "Give solution roadmap in Bengali. Don't solve, only outline steps.",
        "en": "Give solution roadmap. Don't solve, only outline steps."
    },
    "interactive": {
        "bn": "Be an interactive math teacher. Guide student step by step in Bengali.",
        "en": "Be an interactive math teacher. Guide student step by step."
    },
    "quiz": {
        "bn": "Create a 5-question MCQ quiz in Bengali with explanations.",
        "en": "Create a 5-question MCQ quiz with explanations."
    }
}

def get_system_prompt(mode: str, language: str = "bn", difficulty: str = "auto") -> str:
    base = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["detailed"]).get(language, SYSTEM_PROMPTS["detailed"]["en"])
    
    if difficulty != "auto":
        diff_map = {
            "basic": " Keep it simple.",
            "intermediate": " Use intermediate approach.",
            "advanced": " Use advanced methods."
        }
        base += diff_map.get(difficulty, "")
    
    return base

# ---------------------------
# Response Cache
# ---------------------------
class ResponseCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[float, str]] = {}
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> Optional[str]:
        if not CACHE_ENABLED:
            return None
        if key in self.cache:
            timestamp, response = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return response
            del self.cache[key]
        return None

    def set(self, key: str, response: str):
        if CACHE_ENABLED:
            self.cache[key] = (time.time(), response)

response_cache = ResponseCache(ttl_seconds=CACHE_TTL_SECONDS)

# ---------------------------
# AI Provider Calls
# ---------------------------
async def call_groq(question: str, system_prompt: str) -> str:
    """Call Groq API with model fallback."""
    if not GROQ_API_KEY:
        raise ValueError("Groq API key not configured")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    models_to_try = [GROQ_MODEL] + [m for m in GROQ_MODELS if m != GROQ_MODEL]

    for model in models_to_try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        try:
            async with httpx.AsyncClient(timeout=45) as client:
                resp = await client.post(url, json=payload, headers=headers)
                
                if resp.status_code == 404:
                    logger.warning(f"Groq model {model} not found, trying next...")
                    continue
                
                if resp.status_code == 200:
                    data = resp.json()
                    if "choices" in data and data["choices"]:
                        return data["choices"][0]["message"]["content"].strip()
                
                logger.error(f"Groq error: {resp.status_code} - {resp.text[:200]}")
                
        except Exception as e:
            logger.error(f"Groq model {model} error: {str(e)}")

    raise RuntimeError("All Groq models failed")

async def call_gemini(question: str, system_prompt: str) -> str:
    """Call Gemini API."""
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key not configured")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": system_prompt},
                    {"text": question}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048,
        }
    }

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(url, json=payload, headers=headers)
            
            if resp.status_code == 200:
                data = resp.json()
                if "candidates" in data and data["candidates"]:
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            logger.error(f"Gemini error: {resp.status_code} - {resp.text[:200]}")
            raise RuntimeError(f"Gemini API error {resp.status_code}")
            
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        raise

async def call_free_model(question: str, system_prompt: str) -> str:
    """Call free Pollinations AI as last resort."""
    if not FREE_MODEL_ENABLED:
        raise ValueError("Free model disabled")

    url = "https://text.pollinations.ai/openai"
    payload = {
        "model": FREE_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "temperature": 0.7,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"]["content"].strip()
                if content:
                    return content
            raise ValueError("Empty response from free model")
    except Exception as e:
        logger.error(f"Free model error: {str(e)}")
        raise

async def get_ai_answer(question: str, mode: str, language: str = "bn", difficulty: str = "auto") -> str:
    """Try Groq -> Gemini -> Free model."""
    system_prompt = get_system_prompt(mode, language, difficulty)
    errors = []

    cache_key = f"{mode}:{language}:{difficulty}:{question}"
    cached = response_cache.get(cache_key)
    if cached:
        return cached

    if GROQ_API_KEY:
        try:
            logger.info(f"Calling Groq...")
            response = await call_groq(question, system_prompt)
            response_cache.set(cache_key, response)
            return response
        except Exception as e:
            logger.error(f"Groq failed: {e}")
            errors.append(f"Groq: {str(e)}")

    if GEMINI_API_KEY:
        try:
            logger.info(f"Calling Gemini...")
            response = await call_gemini(question, system_prompt)
            response_cache.set(cache_key, response)
            return response
        except Exception as e:
            logger.error(f"Gemini failed: {e}")
            errors.append(f"Gemini: {str(e)}")

    if FREE_MODEL_ENABLED:
        try:
            logger.info(f"Calling free model...")
            response = await call_free_model(question, system_prompt)
            response_cache.set(cache_key, response)
            return response
        except Exception as e:
            logger.error(f"Free model failed: {e}")
            errors.append(f"Free: {str(e)}")

    raise RuntimeError("All providers failed: " + "; ".join(errors))

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/")
async def health_check():
    return {
        "status": "ok",
        "service": "AI Calculation Server",
        "version": "3.0.1",
        "models": {
            "groq": GROQ_MODEL if GROQ_API_KEY else "not configured",
            "gemini": GEMINI_MODEL if GEMINI_API_KEY else "not configured",
            "free_fallback": FREE_MODEL_NAME if FREE_MODEL_ENABLED else "disabled"
        }
    }

@app.get("/api/stats")
async def get_stats(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    return {
        "rate_limit": {
            "limit": RATE_LIMIT_PER_MINUTE,
            "window_seconds": 60
        },
        "cache": {
            "enabled": CACHE_ENABLED,
            "ttl_seconds": CACHE_TTL_SECONDS,
            "entries": len(response_cache.cache)
        },
        "models": {
            "groq": GROQ_MODEL,
            "gemini": GEMINI_MODEL,
            "free_fallback": FREE_MODEL_NAME if FREE_MODEL_ENABLED else "disabled"
        }
    }

@app.post("/api/calculate", dependencies=[Depends(check_rate_limit)])
async def calculate(request: CalculateRequest):
    try:
        start_time = time.time()
        answer = await get_ai_answer(
            request.question,
            request.mode,
            request.language,
            request.difficulty
        )
        processing_time = time.time() - start_time

        return {
            "answer": answer,
            "mode": request.mode,
            "language": request.language,
            "processing_time": f"{processing_time:.2f}s",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Server error")

@app.post("/api/calculate/batch", dependencies=[Depends(check_rate_limit)])
async def calculate_batch(request: BatchCalculateRequest):
    try:
        import asyncio
        tasks = [
            get_ai_answer(q.question, q.mode, q.language, q.difficulty)
            for q in request.questions
        ]
        answers = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for q, answer in zip(request.questions, answers):
            if isinstance(answer, Exception):
                results.append({"question": q.question, "error": str(answer), "success": False})
            else:
                results.append({"question": q.question, "answer": answer, "success": True})

        return {
            "results": results,
            "total": len(results),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"])
        }
    except Exception as e:
        logger.error(f"Batch error: {e}")
        raise HTTPException(status_code=500, detail="Batch processing failed")

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
