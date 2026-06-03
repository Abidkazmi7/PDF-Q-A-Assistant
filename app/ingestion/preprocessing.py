import re

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text