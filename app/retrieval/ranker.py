import re
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from app.retrieval.embedder import model

# Light preprocessing before sentence tokenization
def sentence_clean(text):
    text = re.sub(r'([.?!])([A-Za-z])', r'\1 \2', text)
    return text.strip()

def top_sentences(top_chunks, query_embedding):
    # Obtain index data for each sentence
    sentence_data = []

    for c_idx, chunk in enumerate(top_chunks):
        chunk = sentence_clean(chunk)
        sentences = sent_tokenize(chunk)

        for s_idx, s in enumerate(sentences):
            if len(s.split()) > 5:
                sentence_data.append({
                    "chunk_id": c_idx,
                    "sentences": sentences,
                    "sent_idx": s_idx,
                    "text": s
                })

    # Embed sentences and queries to compare in vector space
    sentence_texts = [x["text"] for x in sentence_data]

    sentence_embeddings = model.encode(
        sentence_texts,
        normalize_embeddings = True
    )

    sentence_similarities = cosine_similarity(
        query_embedding,
        sentence_embeddings
    )[0]

    # Obtain top indices of most similar sentences
    top_indices = np.argsort(sentence_similarities)[-3:][::-1]

    # Obtain results along with their previous and next sentences for context
    expanded_results = []

    for idx in top_indices:
        item = sentence_data[idx]

        sentences = item["sentences"]
        sent_idx = item["sent_idx"]

        prev_sent = sentences[sent_idx - 1] if sent_idx - 1 >= 0 else ""
        curr_sent = sentences[sent_idx]
        next_sent = sentences[sent_idx + 1] if sent_idx + 1 < len(sentences) else ""

        context = " ".join(
            s for s in [prev_sent, curr_sent, next_sent] if s
        )

        expanded_results.append({
            "sentence": context,
            "score": float(sentence_similarities[idx])
        })

    return expanded_results