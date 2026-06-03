import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def chunk_retrieval(query_embedding, chunk_embeddings, chunks, top_k = 5):
    similarities = cosine_similarity(
        query_embedding,
        chunk_embeddings
    )[0]

    top_k = 5

    # Indices of most similar chunks to user query
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [
        {
            "chunk": chunks[i],
            "score": float(similarities[i])
        }
        for i in top_indices
    ]