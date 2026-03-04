"""
BrandSob Agency AI Assistant Backend (FastAPI)

Run:
pip install fastapi uvicorn python-dotenv
uvicorn main:app --reload

.env
GEMINI_API_KEY=your_key_here
"""

import os
import asyncio
import time
from typing import Any, Dict, List, Optional

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None

try:
    from agents import OpenAIChatCompletionsModel, RunConfig, Runner
except Exception:
    OpenAIChatCompletionsModel = None
    RunConfig = None
    Runner = None


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found")


external_client = None
model_wrapper = None
runner = None


if AsyncOpenAI is not None:
    external_client = AsyncOpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


if OpenAIChatCompletionsModel and external_client:

    model_wrapper = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client,
    )

    if RunConfig:
        cfg = RunConfig(
            model=model_wrapper,
            model_provider=external_client,
            tracing_disabled=True
        )

        try:
            runner = Runner(cfg)
        except:
            runner = None


app = FastAPI(title="BrandSob AI Assistant API")


origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


conversations: Dict[str, List[Dict[str, str]]] = {}

MAX_HISTORY_MESSAGES = 40


class ChatRequest(BaseModel):
    user_id: Optional[str] = "default"
    message: str
    system_prompt: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    conversation: List[Dict[str, str]]


def _get_or_create_history(user_id: str):

    history = conversations.get(user_id)

    if history is None:

        history = [
            {
                "role": "system",
                "content": (
                    "You are 'BrandSob AI', the official AI assistant of BrandSob Agency. "
                    "BrandSob is a modern digital agency providing services including:\n"
                    "- Web Development\n"
                    "- AI Agents Development\n"
                    "- Automation Systems\n"
                    "- Digital Marketing\n"
                    "- SaaS Development\n"
                    "- AI Integrations\n\n"

                    "Your job is to help visitors understand BrandSob services, "
                    "answer client questions, explain solutions, and guide potential "
                    "customers to work with BrandSob.\n\n"

                    "Always respond professionally, clearly, and helpfully.\n"

                    "If someone asks unrelated questions not connected to BrandSob "
                    "services or technology, politely redirect the conversation "
                    "toward BrandSob services."
                )
            }
        ]

        conversations[user_id] = history

    return history


def _truncate_history(history, max_msgs=MAX_HISTORY_MESSAGES):

    system_msgs = [m for m in history if m["role"] == "system"]
    non_system = [m for m in history if m["role"] != "system"]

    if len(non_system) > max_msgs:
        non_system = non_system[-max_msgs:]

    history[:] = system_msgs + non_system


async def _call_model(messages):

    if runner:

        try:

            prompt_text = "\n".join(
                [f"{m['role']}: {m['content']}" for m in messages if m["role"] != "system"]
            )

            res = await runner.run(prompt_text)

            if isinstance(res, dict) and "output" in res:
                return str(res["output"])

            return str(res)

        except:
            pass


    if external_client is None:
        raise RuntimeError("No model client configured")


    resp = await external_client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=messages
    )

    return resp.choices[0].message.content



@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):

    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message empty")


    user_id = req.user_id or "default"

    history = _get_or_create_history(user_id)

    history.append({
        "role": "user",
        "content": req.message
    })

    _truncate_history(history)

    try:

        assistant_text = await _call_model(history)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Model error: {e}"
        )


    history.append({
        "role": "assistant",
        "content": assistant_text
    })

    _truncate_history(history)

    return ChatResponse(
        reply=assistant_text,
        conversation=history
    )



@app.get("/")
async def root():
    return {
        "service": "BrandSob AI Assistant",
        "agency": "BrandSob",
        "version": "1.0"
    }