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

# Environment variables (set these in your deployment)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")  # আপডেটেড মডেল
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "500"))
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(
    title="AI Calculation Server - Enhanced",
    description="Advanced stateless AI API for mathematical problem solving with multiple modes",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS (adjust origins as needed)
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
    
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH, description="The math question to solve")
    mode: str = Field(..., pattern="^(detailed|answer_only|roadmap|interactive|quiz)$", description="Response mode")
    language: str = Field("bn", pattern="^(bn|en)$", description="Response language")
    difficulty: str = Field("auto", pattern="^(auto|basic|intermediate|advanced)$", description="Difficulty level")

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        v = re.sub(r"[\x00-\x1f\x7f]", "", v).strip()
        allowed = re.compile(r'[^a-zA-Z0-9\u0980-\u09FF\s\.\,\+\-\*\/\(\)\[\]\{\}\^\=\%\<\>\?\!\:\;\|\\\'\"\@\#\$\&\_\~\`]')
        v = allowed.sub("", v)
        if not v:
            raise ValueError("প্রশ্ন খালি বা অবৈধ অক্ষর রয়েছে")
        if len(v) > MAX_QUESTION_LENGTH:
            raise ValueError(f"প্রশ্ন খুব দীর্ঘ (সর্বোচ্চ {MAX_QUESTION_LENGTH} অক্ষর)")
        return v

class BatchCalculateRequest(BaseModel):
    questions: List[CalculateRequest] = Field(..., min_length=1, max_length=10)
    parallel: bool = Field(True, description="Process questions in parallel")

# ---------------------------
# Advanced Rate Limiter
# ---------------------------
class AdvancedRateLimiter:
    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.blocked_ips: Dict[str, float] = {}
        
    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        
        if ip in self.blocked_ips:
            if now - self.blocked_ips[ip] < 300:
                return False
            else:
                del self.blocked_ips[ip]
        
        self.requests[ip] = [t for t in self.requests[ip] if now - t < self.window_seconds]
        
        if len(self.requests[ip]) >= self.limit:
            if len(self.requests[ip]) >= self.limit * 3:
                self.blocked_ips[ip] = now
                logger.warning(f"IP {ip} blocked for excessive requests")
            return False
        
        self.requests[ip].append(now)
        return True
    
    def get_remaining(self, ip: str) -> int:
        now = time.time()
        self.requests[ip] = [t for t in self.requests[ip] if now - t < self.window_seconds]
        return max(0, self.limit - len(self.requests[ip]))

rate_limiter = AdvancedRateLimiter(limit=RATE_LIMIT_PER_MINUTE)

async def check_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        remaining_time = 60
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429, 
            detail=f"অনেক বেশি রিকোয়েস্ট, {remaining_time} সেকেন্ড পরে আবার চেষ্টা করুন"
        )

# ---------------------------
# Comprehensive System Prompts
# ---------------------------
SYSTEM_PROMPTS = {
    "detailed": {
        "bn": """তুমি একজন অভিজ্ঞ গণিত শিক্ষক এবং সমস্যা সমাধান বিশেষজ্ঞ। তোমার কাজ:

১. প্রশ্নটি মনোযোগ দিয়ে পড়ো এবং বুঝে নাও
২. ধাপে ধাপে সমাধান দাও, প্রতিটি ধাপ ব্যাখ্যা করো
৩. প্রতিটি সূত্র এবং তার প্রয়োগ ব্যাখ্যা করো
৪. সম্ভব হলে একাধিক পদ্ধতি দেখাও
৫. চূড়ান্ত উত্তরটি স্পষ্টভাবে চিহ্নিত করো
৬. সাধারণ ভুল সম্পর্কে সতর্ক করো
৭. প্রয়োজনে চিত্র বা ডায়াগ্রামের বর্ণনা দাও

উত্তরের গঠন:
- সমস্যা বিশ্লেষণ
- প্রয়োজনীয় সূত্র
- ধাপে ধাপে সমাধান
- চূড়ান্ত উত্তর
- যাচাইকরণ (যদি সম্ভব হয়)""",
        
        "en": """You are an experienced mathematics teacher and problem-solving expert. Your task:

1. Read and understand the question carefully
2. Provide step-by-step solution with explanations
3. Explain each formula and its application
4. Show multiple methods when possible
5. Clearly mark the final answer
6. Warn about common mistakes
7. Describe diagrams or figures when needed

Response structure:
- Problem analysis
- Required formulas
- Step-by-step solution
- Final answer
- Verification (if possible)"""
    },
    
    "answer_only": {
        "bn": """শুধুমাত্র চূড়ান্ত উত্তরটি দাও, কোনো ব্যাখ্যা, ধাপ, বা অতিরিক্ত তথ্য ছাড়া। উত্তরটি সংখ্যা, ভগ্নাংশ, বা সংক্ষিপ্ত রাশি হিসেবে দাও।""",
        
        "en": """Provide only the final answer without any explanation, steps, or additional information. Give the answer as a number, fraction, or concise expression."""
    },
    
    "roadmap": {
        "bn": """উত্তর দিয়ো না, শুধু সমাধানের রোডম্যাপ দাও। নিম্নলিখিত কাঠামো অনুসরণ করো:

১. প্রয়োজনীয় সূত্র ও ধারণা
২. ধাপগুলির ক্রম
৩. প্রতিটি ধাপে কী করতে হবে
৪. কোথায় সতর্ক থাকতে হবে
৫. বিকল্প পদ্ধতি (যদি থাকে)""",
        
        "en": """Don't provide the answer, only give a solution roadmap. Follow this structure:

1. Required formulas and concepts
2. Sequence of steps
3. What to do in each step
4. Where to be careful
5. Alternative methods (if any)"""
    },
    
    "interactive": {
        "bn": """তুমি একজন ইন্টারঅ্যাক্টিভ গণিত শিক্ষক। শিক্ষার্থীকে ধাপে ধাপে সমাধান করতে সাহায্য করো:

১. প্রথমে সমস্যাটি বুঝতে সাহায্য করো
২. শিক্ষার্থীকে প্রশ্ন করো
৩. তাদের উত্তর অনুযায়ী পরবর্তী ধাপে যাও
৪. ভুল হলে সংশোধন করতে সাহায্য করো
৫. সঠিক পথে পরিচালিত করো

প্রথমে সমস্যাটি বিশ্লেষণ করো এবং শিক্ষার্থীকে প্রথম প্রশ্নটি করো।""",
        
        "en": """You are an interactive mathematics teacher. Help the student solve step by step:

1. First help understand the problem
2. Ask questions to the student
3. Proceed to next step based on their answers
4. Help correct mistakes
5. Guide in the right direction

First analyze the problem and ask the student the first question."""
    },
    
    "quiz": {
        "bn": """প্রদত্ত বিষয়ের উপর একটি কুইজ তৈরি করো:

১. ৫টি এমসিকিউ প্রশ্ন তৈরি করো
২. প্রতিটি প্রশ্নের ৪টি অপশন দাও
৩. সঠিক উত্তর চিহ্নিত করো
৪. প্রতিটি উত্তরের ব্যাখ্যা দাও
৫. ক্রমবর্ধমান কঠিনতার প্রশ্ন সাজাও

ফরম্যাট:
প্রশ্ন: [প্রশ্ন]
ক) [অপশন]
খ) [অপশন]
গ) [অপশন]
ঘ) [অপশন]
সঠিক উত্তর: [ক/খ/গ/ঘ]
ব্যাখ্যা: [ব্যাখ্যা]""",
        
        "en": """Create a quiz on the given topic:

1. Create 5 MCQ questions
2. Give 4 options for each question
3. Mark the correct answer
4. Explain each answer
5. Arrange questions in increasing difficulty

Format:
Question: [question]
A) [option]
B) [option]
C) [option]
D) [option]
Correct Answer: [A/B/C/D]
Explanation: [explanation]"""
    }
}

def get_system_prompt(mode: str, language: str = "bn", difficulty: str = "auto") -> str:
    """Get system prompt based on mode and language."""
    base_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["detailed"]).get(language, SYSTEM_PROMPTS["detailed"]["bn"])
    
    if difficulty != "auto":
        difficulty_prompt = {
            "basic": "\n\nসমস্যা সহজ রাখো এবং মৌলিক ধারণা ব্যবহার করো।",
            "intermediate": "\n\nমাঝারি মানের সমস্যা সমাধান করো।",
            "advanced": "\n\nউন্নত পদ্ধতি এবং জটিল ধারণা ব্যবহার করো।"
        }
        base_prompt += difficulty_prompt.get(difficulty, "")
    
    return base_prompt

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
                logger.info(f"Cache hit for key: {key[:50]}...")
                return response
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, response: str):
        if CACHE_ENABLED:
            self.cache[key] = (time.time(), response)
            self._cleanup()
    
    def _cleanup(self):
        now = time.time()
        expired_keys = [k for k, (t, _) in self.cache.items() if now - t > self.ttl_seconds]
        for k in expired_keys:
            del self.cache[k]

response_cache = ResponseCache(ttl_seconds=CACHE_TTL_SECONDS)

# ---------------------------
# Enhanced AI Provider Calls
# ---------------------------
async def call_groq(question: str, system_prompt: str) -> str:
    """Call Groq API with enhanced error handling."""
    if not GROQ_API_KEY:
        raise ValueError("Groq API key not configured")
    
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
        "max_tokens": 2048,
        "top_p": 0.9,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,
    }
    
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            
            if "choices" not in data or not data["choices"]:
                raise ValueError("Invalid response format from Groq")
                
            content = data["choices"][0]["message"]["content"].strip()
            if not content:
                raise ValueError("Empty response from Groq")
                
            return content
            
    except httpx.TimeoutException:
        logger.error("Groq API timeout")
        raise TimeoutError("Groq API request timed out")
    except httpx.HTTPStatusError as e:
        logger.error(f"Groq HTTP error: {e.response.status_code} - {e.response.text}")
        raise RuntimeError(f"Groq API returned status {e.response.status_code}")
    except Exception as e:
        logger.error(f"Groq API error: {str(e)}")
        raise

async def call_gemini(question: str, system_prompt: str) -> str:
    """Call Gemini API with new Interactions API."""
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key not configured")
    
    # নতুন Interactions API endpoint
    url = "https://generativelanguage.googleapis.com/v1beta/interactions:create"
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    
    # নতুন Interactions API format
    payload = {
        "model": GEMINI_MODEL,
        "input": f"{system_prompt}\n\n{question}",
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048,
            "topP": 0.9,
            "topK": 40,
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(url, json=payload, headers=headers)
            
            if resp.status_code == 200:
                data = resp.json()
                
                # নতুন response format - output_text ব্যবহার করুন
                if "output_text" in data:
                    content = data["output_text"].strip()
                    if content:
                        return content
                
                # Fallback: পুরনো format check
                if "candidates" in data and data["candidates"]:
                    content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    if content:
                        return content
                
                raise ValueError("No valid response from Gemini")
                
            elif resp.status_code == 404:
                raise RuntimeError(f"Gemini model {GEMINI_MODEL} not found")
            elif resp.status_code == 429:
                raise RuntimeError("Gemini rate limit exceeded")
            else:
                error_detail = resp.text
                raise RuntimeError(f"Gemini API error {resp.status_code}: {error_detail}")
                
    except httpx.TimeoutException:
        logger.error("Gemini API timeout")
        raise TimeoutError("Gemini API request timed out")
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        raise

async def get_ai_answer(question: str, mode: str, language: str = "bn", difficulty: str = "auto") -> str:
    """Try multiple AI providers with fallback."""
    system_prompt = get_system_prompt(mode, language, difficulty)
    errors = []
    
    # Generate cache key
    cache_key = f"{mode}:{language}:{difficulty}:{question}"
    
    # Check cache first
    cached_response = response_cache.get(cache_key)
    if cached_response:
        return cached_response
    
    # Try Groq first (faster)
    if GROQ_API_KEY:
        try:
            logger.info(f"Calling Groq with mode={mode}, language={language}")
            response = await call_groq(question, system_prompt)
            response_cache.set(cache_key, response)
            return response
        except Exception as e:
            logger.error(f"Groq failed: {e}")
            errors.append(f"Groq: {str(e)}")
    
    # Try Gemini as fallback
    if GEMINI_API_KEY:
        try:
            logger.info(f"Calling Gemini with mode={mode}, language={language}")
            response = await call_gemini(question, system_prompt)
            response_cache.set(cache_key, response)
            return response
        except Exception as e:
            logger.error(f"Gemini failed: {e}")
            errors.append(f"Gemini: {str(e)}")
    
    # If both fail, raise error
    error_msg = "AI providers failed: " + "; ".join(errors) if errors else "No AI providers configured"
    raise RuntimeError(error_msg)

# ---------------------------
# Enhanced Endpoints
# ---------------------------
@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "AI Calculation Server",
        "version": "3.0.0",
        "models": {
            "groq": GROQ_MODEL if GROQ_API_KEY else "not configured",
            "gemini": GEMINI_MODEL if GEMINI_API_KEY else "not configured"
        }
    }

@app.get("/api/stats")
async def get_stats(request: Request):
    """Get server statistics."""
    client_ip = request.client.host if request.client else "unknown"
    remaining = rate_limiter.get_remaining(client_ip)
    
    return {
        "rate_limit": {
            "limit": RATE_LIMIT_PER_MINUTE,
            "remaining": remaining,
            "window_seconds": 60
        },
        "cache": {
            "enabled": CACHE_ENABLED,
            "ttl_seconds": CACHE_TTL_SECONDS,
            "entries": len(response_cache.cache)
        },
        "models": {
            "groq": GROQ_MODEL,
            "gemini": GEMINI_MODEL
        }
    }

@app.post("/api/calculate", dependencies=[Depends(check_rate_limit)])
async def calculate(request: CalculateRequest):
    """
    Process a math question with the specified mode and return the AI answer.
    
    Modes:
    - detailed: Step-by-step solution with explanations
    - answer_only: Just the final answer
    - roadmap: Solution roadmap without the answer
    - interactive: Interactive teaching approach
    - quiz: Generate a quiz on the topic
    """
    try:
        start_time = time.time()
        answer = await get_ai_answer(
            request.question, 
            request.mode, 
            request.language, 
            request.difficulty
        )
        processing_time = time.time() - start_time
        
        logger.info(f"Question processed in {processing_time:.2f}s")
        
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
        raise HTTPException(
            status_code=500, 
            detail="সার্ভারে ত্রুটি হয়েছে, পরে আবার চেষ্টা করুন"
        )

@app.post("/api/calculate/batch", dependencies=[Depends(check_rate_limit)])
async def calculate_batch(request: BatchCalculateRequest):
    """Process multiple math questions in batch."""
    try:
        if request.parallel:
            import asyncio
            tasks = [
                get_ai_answer(
                    q.question, 
                    q.mode, 
                    q.language, 
                    q.difficulty
                ) 
                for q in request.questions
            ]
            answers = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            answers = []
            for q in request.questions:
                try:
                    answer = await get_ai_answer(
                        q.question, 
                        q.mode, 
                        q.language, 
                        q.difficulty
                    )
                    answers.append(answer)
                except Exception as e:
                    answers.append(f"Error: {str(e)}")
        
        results = []
        for i, (q, answer) in enumerate(zip(request.questions, answers)):
            if isinstance(answer, Exception):
                results.append({
                    "question": q.question,
                    "error": str(answer),
                    "success": False
                })
            else:
                results.append({
                    "question": q.question,
                    "answer": answer,
                    "success": True
                })
        
        return {
            "results": results,
            "total": len(results),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"])
        }
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(status_code=500, detail="ব্যাচ প্রসেসিং ব্যর্থ হয়েছে")

# ---------------------------
# Error Handlers
# ---------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# ---------------------------
# Run (for local development)
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        reload=True
    )
