"""
Streamlit UI: enter a movie name in the prompt box and get a sentiment prediction.
"""

import os
import sys
import streamlit as st

# Make src importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.predict import predict_movie, load_artifacts  # noqa: E402


st.set_page_config(page_title="Telugu Movie Sentiment Analyzer", page_icon="🎬", layout="centered")

st.title("🎬 Telugu Movie Sentiment Analyzer")
st.caption("Naive Bayes classifier trained on Telugu hit / flop movie data")

# Ensure model exists
if not (os.path.exists("model/nb_model.pkl") and os.path.exists("model/vectorizer.pkl")):
    st.error("⚠️ Model not found. Please run `python src/train_model.py` first.")
    st.stop()

movie_name = st.text_input("Enter a movie name", placeholder="e.g. Bahubali")
extra_review = st.text_area(
    "Optional: paste a review snippet (helps the model when the movie is not in the dataset)",
    placeholder="e.g. Great visuals and powerful performances",
    height=100,
)

if st.button("🔍 Predict Sentiment"):
    if not movie_name.strip():
        st.warning("Please enter a movie name.")
    else:
        result = predict_movie(movie_name.strip(), extra_review.strip())

        if result["sentiment"] == "positive":
            st.success(f"✅ **{result['movie_name']}** is a **HIT (Positive)**")
        else:
            st.error(f"❌ **{result['movie_name']}** is a **FLOP (Negative)**")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", result["sentiment"].capitalize())
        with col2:
            st.metric("Verdict", result["verdict"].capitalize())

        if "stars" in result:
            st.write(f"⭐ **Stars:** {result['stars']}")
        if "confidence" in result:
            st.write(f"📊 **Confidence:** {result['confidence']}%")
        if "review" in result:
            st.write(f"📝 **Review:** {result['review']}")

        st.caption(f"Prediction source: {result['source']}")