"""
Load the trained model and predict sentiment for a given movie name / review.
"""

import os
import sys
import joblib
import pandas as pd

from preprocess import clean_text


MODEL_PATH = os.path.join("model", "nb_model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")
DATA_PATH = os.path.join("data", "telugu_movies.csv")


def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        raise FileNotFoundError(
            "Trained model not found. Run `python src/train_model.py` first."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def lookup_known_movie(movie_name: str):
    """Check if the movie exists in the dataset (exact or partial match)."""
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    name = movie_name.strip().lower()
    match = df[df["movie_name"].str.lower() == name]
    if match.empty:
        match = df[df["movie_name"].str.lower().str.contains(name, regex=False)]
    if not match.empty:
        row = match.iloc[0]
        return {
            "movie_name": row["movie_name"],
            "verdict": row["verdict"],
            "sentiment": row["sentiment"],
            "stars": row["stars"],
            "review": row["review"],
            "source": "dataset",
        }
    return None


def predict_movie(movie_name: str, extra_review: str = "") -> dict:
    """
    Predict sentiment for a movie. If movie is in dataset, return its known label;
    otherwise use the trained Naive Bayes model on the name + optional review.
    """
    known = lookup_known_movie(movie_name)
    if known is not None and not extra_review:
        return known

    model, vectorizer = load_artifacts()
    text = clean_text(f"{movie_name} {extra_review}".strip())
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    confidence = proba[classes.index(pred)]

    return {
        "movie_name": movie_name,
        "sentiment": pred,
        "verdict": "hit" if pred == "positive" else "flop",
        "confidence": round(float(confidence) * 100, 2),
        "source": "model",
    }


def main():
    if len(sys.argv) > 1:
        movie = " ".join(sys.argv[1:])
    else:
        movie = input("🎬 Enter a movie name: ").strip()

    result = predict_movie(movie)
    print("\n" + "=" * 50)
    print(f"Movie     : {result['movie_name']}")
    print(f"Sentiment : {result['sentiment'].upper()}")
    print(f"Verdict   : {result['verdict'].upper()}")
    if "stars" in result:
        print(f"Stars     : {result['stars']} ⭐")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']}%")
    if "review" in result:
        print(f"Review    : {result['review']}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()