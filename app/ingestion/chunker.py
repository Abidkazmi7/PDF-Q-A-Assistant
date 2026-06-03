from nltk.tokenize import word_tokenize

def create_chunks(text, chunk_size = 200, overlap = 50):
    words = word_tokenize(text)
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)

    return chunks