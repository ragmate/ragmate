import json
import threading
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncGenerator

from fastapi import Depends, FastAPI, Request
from fastapi.responses import PlainTextResponse, StreamingResponse

from .config import Settings, get_settings
from .llm import LLM
from .models import ChatRequestModel
from .services import DirectoryMonitorService, VectorStoreService


@asynccontextmanager
async def lifespan(app: FastAPI):
    vector_store_service = VectorStoreService()
    directory_monitor_service = DirectoryMonitorService(vector_store=vector_store_service)

    app.llm = LLM(vector_store=vector_store_service)  # type: ignore[attr-defined]
    threading.Thread(target=directory_monitor_service.watch_directory, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return PlainTextResponse("Ollama is running")


@app.get("/api/tags")
async def tags(settings: Annotated[Settings, Depends(get_settings)]):
    return {
        "models": [
            {
                "name": settings.LOCAL_MODEL,
                "model": settings.LOCAL_MODEL,
            }
        ]
    }


@app.post("/api/chat")
async def chat(body: ChatRequestModel, request: Request, settings: Annotated[Settings, Depends(get_settings)]):

    async def event_generator() -> AsyncGenerator[bytes, Any]:
        async for resp in request.app.llm.chat(messages=body.messages):
            if isinstance(resp, dict):
                if "llm" not in resp:
                    continue
                content = resp["llm"]["answer"].content
            else:
                content = resp

            data = {
                "model": settings.LOCAL_MODEL,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "done": False,
            }
            yield (json.dumps(data) + "\n").encode()

        final = {
            "model": settings.LOCAL_MODEL,
            "message": {"role": "assistant", "content": ""},
            "done_reason": "stop",
            "done": True,
        }
        yield (json.dumps(final) + "\n").encode()

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
