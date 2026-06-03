from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.ingestion.pdf_loader import extract_text
from app.ingestion.chunker import create_chunks
from app.retrieval.embedder import embed_chunks, embed_query
from app.retrieval.retriever import chunk_retrieval
from app.retrieval.ranker import top_sentences

app = FastAPI()

chunks = None
chunk_embeddings = None

# Request model
class QueryRequest(BaseModel):
    question: str

# Load PDF
@app.post("/load-pdf")
def load_pdf(pdf_path: str):
    global chunks, chunk_embeddings

    text = extract_text(pdf_path)
    chunks = create_chunks(text)
    chunk_embeddings = embed_chunks(chunks)

    return {
        "message": "PDF loaded successfully",
        "chunks": len(chunks)
    }

# Query endpoint
@app.post("/query")
def query(req: QueryRequest):
    if chunks is None or chunk_embeddings is None:
        raise HTTPException(status_code = 400, detail="No PDF loaded")
    
    query_embedding = embed_query([req.question])

    top_chunk_results = chunk_retrieval(
        query_embedding,
        chunk_embeddings,
        chunks,
        top_k = 3
    )

    top_chunks = [r["chunk"] for r in top_chunk_results]

    sentence_results = top_sentences(
        top_chunks,
        query_embedding
    )

    return {
        "query": req.question,
        "top_chunks": top_chunks,
        "answer_sentences": sentence_results
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "pdf_loaded": chunks is not None
    }