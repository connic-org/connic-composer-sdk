from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import tool


@dataclass
class RetrievedDoc:
    metadata: dict
    page_content: str


class FakeVectorStore:
    def similarity_search(self, query: str, k: int = 2) -> list[RetrievedDoc]:
        docs = [
            RetrievedDoc(
                metadata={"source": "internal://rag-doc-1"},
                page_content="Task decomposition breaks a large task into smaller, manageable steps.",
            ),
            RetrievedDoc(
                metadata={"source": "internal://rag-doc-2"},
                page_content="Common extensions include chain-of-thought, tree-of-thought, and plan-and-execute approaches.",
            ),
        ]
        return docs[:k]


vector_store = FakeVectorStore()


@tool
def retrieve_context(query: str) -> str:
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    return "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in retrieved_docs
    )


rag_prompt = (
    "You have access to a tool backed by managed semantic retrieval. "
    "Use the tool to help answer user queries. "
    "If the retrieved context does not contain relevant information to answer "
    "the query, say that you don't know."
)


agent = create_agent(
    model="openai:gpt-5.2",
    tools=[retrieve_context],
    system_prompt=rag_prompt,
)
