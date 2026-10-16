"""
Text preprocessing utilities: tokenization + vectorization.
"""

import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, digits and extra whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)      # URLs
    text = re.sub(r"\d+", " ", text)                    # digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    """Simple whitespace tokenizer after cleaning."""
    return clean_text(text).split()


def build_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """Return a configured TF-IDF vectorizer."""
    return TfidfVectorizer(
        preprocessor=clean_text,
        tokenizer=tokenize,
        token_pattern=None,
        ngram_range=(1, 2),
        max_features=max_features,
    )


if __name__ == "__main__":
    sample = "Bahubali was an EPIC masterpiece!!! 100% worth watching."
    print("Cleaned :", clean_text(sample))
    print("Tokens  :", tokenize(sample))