import os
from typing import TypedDict, Literal

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

MOCK_LLM = os.getenv("MOCK_LLM", "1")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class GraphState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved_docs: list
    response: dict


def classify_intent(state: GraphState):
    query = state["query"].lower()

    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours"
    ]

    if any(keyword in query for keyword in policy_keywords):
        intent = "policy_question"
    else:
        intent = "general_question"

    return {"intent": intent}


def retrieve_and_answer(state: GraphState):
    query = state["query"]

    results = vectorstore.similarity_search_with_score(
        query,
        k=3
    )

    documents = [item[0] for item in results]

    if not documents:
        response = AnswerResponse(
            answer="I could not find relevant information in the Zepto policy documents.",
            sources=[],
            confidence=0.0
        )
        return {
            "retrieved_docs": [],
            "response": response.model_dump()
        }

    top_document = documents[0]

    snippet = top_document.page_content[:200].strip()

    source_ids = [
        doc.metadata.get("source", "unknown")
        for doc in documents
    ]

    if MOCK_LLM == "1":
        answer = f"Based on the retrieved context: {snippet}"

        response = AnswerResponse(
            answer=answer,
            sources=source_ids,
            confidence=1.0
        )

        return {
            "retrieved_docs": documents,
            "response": response.model_dump()
        }

    answer = f"Based on the retrieved context: {snippet}"

    response = AnswerResponse(
        answer=answer,
        sources=source_ids,
        confidence=1.0
    )

    return {
        "retrieved_docs": documents,
        "response": response.model_dump()
    }


def direct_answer(state: GraphState):
    response = AnswerResponse(
        answer="I can only answer questions about Zepto policies right now.",
        sources=[],
        confidence=1.0
    )

    return {
        "response": response.model_dump()
    }


def route_after_classification(
    state: GraphState
):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


workflow = StateGraph(GraphState)

workflow.add_node(
    "classify_intent",
    classify_intent
)

workflow.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

workflow.add_node(
    "direct_answer",
    direct_answer
)

workflow.set_entry_point("classify_intent")

workflow.add_conditional_edges(
    "classify_intent",
    route_after_classification,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

workflow.add_edge(
    "retrieve_and_answer",
    END
)

workflow.add_edge(
    "direct_answer",
    END
)

graph = workflow.compile()


if __name__ == "__main__":
    policy_result = graph.invoke({
        "query": "What is the refund policy?"
    })

    print("\nPOLICY QUESTION")
    print(policy_result["response"])

    general_result = graph.invoke({
        "query": "What is the capital of India?"
    })

    print("\nGENERAL QUESTION")
    print(general_result["response"])