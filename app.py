import streamlit as st
import pickle
import pandas as pd
import numpy as np

@st.cache_resource(show_spinner=False)
def load_data():
    movie_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movie_dict)
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    similarity = np.array(similarity)  # ensure numpy array
    return movies, similarity

movies, similarity = load_data()

st.title("🎬 Movie Recommender System")

selected_movie_name = st.selectbox(
    "Select a movie",
    sorted(movies['title'].dropna().unique().tolist())
)

def recommend(movie_title: str, top_n: int = 5):
    # robust lookup: get list of matching indices (handles duplicates)
    matches = movies.index[movies['title'] == movie_title].tolist()
    if not matches:
        return [], []
    movie_index = matches[0]

    # defensive checks
    if similarity.shape[0] != len(movies):
        # similarity matrix likely misaligned with movies
        return [], []

    distances = list(enumerate(similarity[movie_index]))
    # sort by score descending, skip index 0 (the movie itself)
    distances = sorted(distances, key=lambda x: x[1], reverse=True)[1: top_n + 1]

    recommended_titles = []
    debug_info = []
    for idx, score in distances:
        recommended_titles.append(movies.iloc[idx]['title'])
        debug_info.append((idx, float(score), movies.iloc[idx]['title']))
    return recommended_titles, debug_info

if st.button("Recommend"):
    recs, info = recommend(selected_movie_name, top_n=5)
    if not recs:
        st.warning("No recommendations found — see debug below.")
    else:
        st.subheader("You may also like:")
        for t in recs:
            st.write("•", t)

    if st.checkbox("Show debug info"):
        st.write("movies.shape:", movies.shape)
        st.write("similarity.shape:", similarity.shape)
        st.write("recommend debug (index, score, title):", info)
