import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Cache data and model loading
# -------------------------------

@st.cache_resource
def load_model_data():
    with open("book_recommendation_model.pkl", "rb") as file:
        model_data = pickle.load(file)
    return model_data

# Load model data once
model_data = load_model_data()

movie_vectors = model_data["movie_vectors"]
book_vectors = model_data["book_vectors"]
movie_dataset = model_data["movie_dataset"]
books_dataset = model_data["books_dataset"]

# Add lowercase column for case-insensitive matching
movie_dataset["title_lower"] = movie_dataset["title"].str.lower()

# -------------------------------
# Recommendation Logic
# -------------------------------

@st.cache_data
def recommend_books(movie_title, top_n=5):
    movie_title = movie_title.lower()

    if movie_title not in movie_dataset["title_lower"].values:
        return []

    # Get index and movie vector
    movie_index = movie_dataset[movie_dataset["title_lower"] == movie_title].index[0]
    movie_vector = movie_vectors[movie_index]

    # Compute cosine similarity
    similarity_scores = cosine_similarity(movie_vector.reshape(1, -1), book_vectors)[0]
    top_indices = np.argsort(similarity_scores)[-top_n:][::-1]

    # Get top book entries
    top_books = books_dataset.iloc[top_indices]

    recommendations = []
    for _, book in top_books.iterrows():
        recommendations.append({
            "title": book["title"],
            "author": book["author"],
            "coverImg": book["coverImg"],
            "ratings": book["rating"],
            "awards": book["awards"] if isinstance(book["awards"], list) and book["awards"] else None
        })

    return recommendations

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="Book Recommendations", layout="wide")
st.title("📚 Book Recommendations Based on Your Favorite Movie 🎬")

st.markdown("Pick a movie and discover books you might enjoy!")

# Movie selector
movie_list = sorted(movie_dataset["title"].tolist())
selected_movie = st.selectbox("🎥 Select a movie:", movie_list)

# Recommend button
if st.button("🔍 Recommend Books"):
    results = recommend_books(selected_movie)

    if results:
        st.success(f"Showing top recommendations based on *{selected_movie}*")
        for book in results:
            with st.expander(f"📖 {book['title']}"):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(book["coverImg"], use_column_width=True)
                with col2:
                    st.markdown(f"**Author:** {book['author']}")
                    st.markdown(f"**Rating:** {book['ratings']}")
                    if book["awards"]:
                        st.markdown("**Awards:**")
                        for award in book["awards"]:
                            st.markdown(f"- {award}")
    else:
        st.warning("❌ Movie not found in the dataset. Please try another title.")
