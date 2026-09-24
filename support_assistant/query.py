from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=str(DB_DIR),
    embedding_function=embeddings
)

question = input("Enter your question: ")

results = vectorstore.similarity_search(question, k=3)

print("\nRelevant information:\n")

for i, result in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(f"Source: {result.metadata.get('source', 'Unknown')}")
    print(result.page_content)
    print()