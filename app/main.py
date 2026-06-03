from ingestion.pdf_loader import extract_text
from ingestion.chunker import create_chunks
from retrieval.embedder import embed_chunks, embed_query
from retrieval.retriever import chunk_retrieval
from retrieval.ranker import top_sentences

def build_index(pdf_path):
    # 1. Load PDF
    text = extract_text(pdf_path)

    # 2. Chunk text
    chunks = create_chunks(text)

    # 3. Embed chunks
    chunk_embeddings = embed_chunks(chunks)

    return chunks, chunk_embeddings

def ask_question(query, chunks, chunk_embeddings):
    # 1. Embed query
    query_embedding = embed_query([query])

    # 2. Retrieve top chunks
    top_chunk_results = chunk_retrieval(
        query_embedding,
        chunk_embeddings,
        chunks,
        top_k = 5
    )

    top_chunks = [r["chunk"] for r in top_chunk_results]

    # 3. Sentence-level ranking (final answer extraction)
    sentence_results = top_sentences(
        top_chunks,
        query_embedding
    )

    return{
        "query" : query,
        "top_chunks" : top_chunks,
        "answer_sentences" : sentence_results
    }

if __name__ == "__main__":
    pdf_path = r"E:\Machine Learning\Projects\PDF Q&A Assistant\data\Stanford ML Notes.pdf"

    chunks, chunk_embeddings = build_index(pdf_path)

    while True:
        query = input("Ask: ")

        result = ask_question(
            query,
            chunks,
            chunk_embeddings
        )

        print("\nTOP CHUNKS:\n")
        for c in result["top_chunks"]:
            print(c[:300], "\n---")

        print("\nANSWER SENTENCES:\n")
        for s in result["answer_sentences"]:
            print(s["sentence"])
            print("Context:", s["previous"], "|", s["next"])
            print("Score:", s["score"])
            print("---")