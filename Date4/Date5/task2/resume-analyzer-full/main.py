import os
import json
import asyncio
import logging
from uuid import uuid4
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from openai import OpenAI, RateLimitError

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError('OPENAI_API_KEY not found in .env file')

MAX_RESUME_TOKENS = 5000
LLM_TIMEOUT = 15
MAX_RETRIES = 3
BATCH_CONCURRENCY_LIMIT = 5

app = FastAPI(title='Resume Analyzer API', version='1.0.0')
client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class ResumeRequest(BaseModel):
    resume: str

class BatchResumeRequest(BaseModel):
    resumes: List[str]

def estimate_tokens(text:str)->int:
    return len(text.split())

async def call_llm(resume_text:str, request_id:str):
    for attempt in range(MAX_RETRIES):
        try:
            response = await asyncio.wait_for(asyncio.to_thread(
                client.chat.completions.create,
                model='gpt-4o-mini',
                messages=[{'role':'system','content':'Return JSON only.'},{'role':'user','content':resume_text}]
            ), timeout=LLM_TIMEOUT)
            result = response.choices[0].message.content
            return json.loads(result)
        except asyncio.TimeoutError:
            if attempt == MAX_RETRIES-1:
                raise HTTPException(status_code=503, detail='The AI service is taking longer than expected. Please try again later.')
        except RateLimitError:
            if attempt == MAX_RETRIES-1:
                raise HTTPException(status_code=503, detail='Too many requests currently. Please try again shortly.')
        except Exception:
            if attempt == MAX_RETRIES-1:
                raise HTTPException(status_code=502, detail='Something went wrong while processing your resume.')

@app.get('/')
def root():
    return {'status':'healthy','service':'Resume Analyzer API'}

@app.post('/analyze')
async def analyze(payload: ResumeRequest, request: Request):
    request_id = str(uuid4())
    tokens = estimate_tokens(payload.resume)
    if tokens > MAX_RESUME_TOKENS:
        raise HTTPException(status_code=413, detail='Resume exceeds maximum allowed size.')
    return {'request_id':request_id,'analysis':await call_llm(payload.resume, request_id)}

semaphore = asyncio.Semaphore(BATCH_CONCURRENCY_LIMIT)
async def process_resume(resume:str):
    async with semaphore:
        rid = str(uuid4())
        try:
            return {'request_id':rid,'status_code':200,'analysis':await call_llm(resume,rid)}
        except HTTPException as e:
            return {'request_id':rid,'status_code':e.status_code,'message':e.detail}

@app.post('/analyze-batch')
async def analyze_batch(payload: BatchResumeRequest):
    results = await asyncio.gather(*[process_resume(r) for r in payload.resumes])
    return {'results':results}
