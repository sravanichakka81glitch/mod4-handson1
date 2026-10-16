"""
Train a Naive Bayes sentiment classifier on the Telugu movie dataset.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import build_vectorizer, clean_text


DATA_PATH = os.path.join("data", "telugu_movies.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "nb_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")


def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Combine name + review so the model learns both movie name and review text
    df["text"] = (df["movie_name"].fillna("") + " " + df["review"].fillna("")).apply(clean_text)
    return df


def train():
    df = load_and_prepare(DATA_PATH)
    print(f"📦 Loaded {len(df)} records.")
    print("Label distribution:\n", df["sentiment"].value_counts(), "\n")

    X_raw = df["text"].values
    y = df["sentiment"].values

    # 80/20 stratified split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"🎯 Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\n💾 Model saved to       : {MODEL_PATH}")
    print(f"💾 Vectorizer saved to  : {VECTORIZER_PATH}")


if __name__ == "__main__":
    train()