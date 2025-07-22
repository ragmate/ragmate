from typing import Any

from pydantic import BaseModel


class ChatMessageModel(BaseModel):
    role: str
    content: str


class ChatRequestModel(BaseModel):
    model: str
    messages: list[ChatMessageModel]
    stream: bool
    keep_alive: str
    options: dict[str, Any]


class CheckoutEventModel(BaseModel):
    previous_head: str
    new_head: str
