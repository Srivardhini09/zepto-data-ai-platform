from fastapi import FastAPI
from pydantic import BaseModel

from support_assistant.graph import graph


app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-based Zepto policy support assistant",
    version="1.0.0"
)


class AskRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {
        "message": "Zepto Support Assistant API is running"
    }


@app.post("/ask")
def ask(request: AskRequest):
    result = graph.invoke({
        "query": request.query
    })

    return result["response"]