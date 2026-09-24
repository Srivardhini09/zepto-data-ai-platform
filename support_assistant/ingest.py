from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "chroma_db"

documents = []

for file_path in sorted(DOCS_DIR.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8")

    documents.append(
        Document(
            page_content=text,
            metadata={"source": file_path.name}
        )
    )

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(DB_DIR)
)

print(f"Loaded {len(documents)} documents.")
print(f"Created {len(chunks)} chunks.")
print(f"Vector database saved to: {DB_DIR}")