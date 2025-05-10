from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

from .config import get_settings
from .models import ChatMessageModel

if TYPE_CHECKING:
    from langchain_core.runnables.base import Runnable

    from .services import VectorStoreService


# set_debug(True)


class State(TypedDict):
    messages: list["ChatMessageModel"]
    prompt: list["ChatMessageModel"]
    context: list[Document]
    answer: str


class LLM:
    files: list[str] = []

    def __init__(self, vector_store: "VectorStoreService") -> None:
        self.vector_store = vector_store

    @staticmethod
    async def _format_context(docs: list["Document"]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    async def _build_prompt(self, messages: list[ChatMessageModel], docs: list[Document]):
        messages = list(messages)
        framework = get_settings().FRAMEWORK
        if framework is not None:
            messages.append(
                ChatMessageModel(
                    role="user",
                    content=f"**Context:** The project code based on the framework {framework}. Be sure to take this into account.",
                )
            )
        if docs:
            formatted_docs = await self._format_context(docs=docs)
            messages.append(
                ChatMessageModel(role="user", content=f"Attached files (extracted text):\n{formatted_docs}")
            )
        return messages

    async def _build_prompt_node(self, state: State) -> dict[str, Any]:
        retriever = self.vector_store.get_retriever()
        query = ""
        for message in state["messages"]:
            query += message.content + "\n\n"

        docs = await retriever.ainvoke(input=query)
        prompt = await self._build_prompt(messages=state["messages"], docs=docs)
        return {
            "prompt": prompt,
        }

    @staticmethod
    async def to_langchain_messages(messages: list[ChatMessageModel]) -> list[BaseMessage]:
        converted: list[BaseMessage] = []
        for msg in messages:
            if msg.role == "user":
                converted.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                converted.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                converted.append(SystemMessage(content=msg.content))
            else:
                raise ValueError(f"Unknown role: {msg.role}")
        return converted

    async def _llm_node_func(self, state: State) -> dict[str, Any]:
        llm = init_chat_model(
            get_settings().OPENAI_MODEL, model_provider="openai", api_key=get_settings().OPENAI_API_KEY
        )
        converted_prompt = await self.to_langchain_messages(messages=state["prompt"])
        return {"answer": await llm.ainvoke(converted_prompt)}

    async def chat(self, messages: list["ChatMessageModel"]) -> AsyncGenerator[dict[str, Any], None]:
        prompt_node: "Runnable[State, Awaitable[dict[str, Any]]]" = RunnableLambda(self._build_prompt_node)

        llm_node: "Runnable[State, Awaitable[dict[str, Any]]]" = RunnableLambda(self._llm_node_func)

        graph = StateGraph(state_schema=State)
        graph.add_node("prompt_builder", prompt_node)
        graph.add_node("llm", llm_node)

        graph.set_entry_point("prompt_builder")
        graph.add_edge("prompt_builder", "llm")
        graph.set_finish_point("llm")

        chain = graph.compile()

        async for chunk in chain.astream({"messages": messages}, stream_mode="updates"):
            yield chunk
