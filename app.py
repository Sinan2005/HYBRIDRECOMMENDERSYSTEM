# app.py
import streamlit as st
from recommender import (
    movies,
    recommend_movies_content,
    rank_movies_cf,
    generate_rich_explanation
)

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")
st.caption("Hybrid Recommender • Content-Based + Collaborative Filtering")

movie_list = movies[["movieId", "title"]].sort_values("title")

selected_title = st.selectbox(
    "Select a movie",
    movie_list["title"]
)

user_id = st.number_input("User ID", min_value=1, value=10)
top_n = st.slider("Number of recommendations", 5, 20, 10)

movie_id = movie_list[movie_list["title"] == selected_title]["movieId"].values[0]

if st.button("Recommend"):
    with st.spinner("Finding movies you’ll love..."):
        content_recs = recommend_movies_content(movie_id, top_n)

        ranked = rank_movies_cf(
            user_id,
            content_recs["movieId"].tolist(),
            top_n
        )

        st.subheader("Recommended Movies")

        for mid, score in ranked:
            title = movie_list[movie_list.movieId == mid]["title"].values[0]
            explanation = generate_rich_explanation(movie_id, mid)

            st.markdown(f"### 🎥 {title}")
            st.write(f"⭐ Predicted Rating: {score:.2f}")
            st.info(explanation)
            st.divider()
