from typing import TYPE_CHECKING, Any, AsyncGenerator

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

from .config import get_settings
from .models import ChatMessageModel

if TYPE_CHECKING:
    from .services import VectorStoreService


# set_debug(True)


class State(TypedDict):
    messages: list["ChatMessageModel"]
    prompt: list["ChatMessageModel"]
    context: list[Document]
    answer: str


class LLM:
    files: list[str] = []
    settings = get_settings()

    def __init__(self, vector_store: "VectorStoreService") -> None:
        self.vector_store = vector_store

    @staticmethod
    async def _format_context(docs: list["Document"]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    async def _build_prompt(self, messages: list[ChatMessageModel], docs: list[Document]):
        framework = self.settings.FRAMEWORK
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
        if any(
            item_a in item_b.content
            for item_a in self.settings.SKIP_RAG_FOR_CHAT_THAT_CONTAINS
            for item_b in state["messages"]
        ):
            return {"prompt": state["messages"]}

        retriever = self.vector_store.get_retriever()
        retriever_input = await self.prepare_retriever_input(user_prompt=state["messages"][-1].content)
        docs = await retriever.ainvoke(input=retriever_input)
        prompt = await self._build_prompt(messages=state["messages"], docs=docs)
        return {
            "prompt": prompt,
        }

    async def prepare_retriever_input(self, user_prompt: str) -> str:
        query_prompt = """
You are an assistant that converts long user prompts into short, clear retrieval queries for information search (retrieval).

Below is a user prompt:
'''
{user_prompt}
'''

Your task:
Extract and return only the final retrieval query, formulated as clearly and concisely as possible, so that it can be used to search for relevant code or documents.

Return only the query text, without any explanations or additional comments.
            """
        response = await self.run(query=query_prompt.format(user_prompt=user_prompt))
        return str(response.content)

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

    async def _llm_node(self, state: State) -> dict[str, BaseMessage]:
        converted_prompt = await self.to_langchain_messages(messages=state["prompt"])
        return {"answer": await self.run(query=converted_prompt)}

    async def run(self, query: str | list[BaseMessage]) -> BaseMessage:
        llm = init_chat_model(
            self.settings.LLM_MODEL,
            model_provider=self.settings.LLM_PROVIDER,
            temperature=self.settings.LLM_TEMPERATURE,
            base_url=self.settings.LLM_BASE_URL,
            api_key=self.settings.LLM_API_KEY,
        )
        response = await llm.ainvoke(query)
        return response

    async def chat(self, messages: list["ChatMessageModel"]) -> AsyncGenerator[dict[str, Any], None]:
        graph = StateGraph(state_schema=State)
        graph.add_node("prompt_builder", self._build_prompt_node)
        graph.add_node("llm", self._llm_node)

        graph.set_entry_point("prompt_builder")
        graph.add_edge("prompt_builder", "llm")
        graph.set_finish_point("llm")

        chain = graph.compile()

        async for chunk in chain.astream({"messages": messages}, stream_mode="updates"):
            yield chunk
