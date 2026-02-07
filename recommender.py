# recommender.py
import pandas as pd
import pickle
import ast

# -----------------------------
# LOAD DATA
# -----------------------------
movies = pd.read_csv(
    "data/movies.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["movieId", "title", "genres"]
)

ratings = pd.read_csv(
    "data/ratings.dat",
    sep="::",
    engine="python",
    names=["userId", "movieId", "rating", "timestamp"]
)

credits = pd.read_csv("data/credits.csv")
keywords = pd.read_csv("data/keywords.csv")
metadata = pd.read_csv("data/movies_metadata.csv")

# -----------------------------
# LOAD MODELS
# -----------------------------
with open("models/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/cosine_sim.pkl", "rb") as f:
    cosine_sim = pickle.load(f)

with open("models/svd_model.pkl", "rb") as f:
    svd = pickle.load(f)

# -----------------------------
# HELPERS
# -----------------------------
def get_cast(cast_str, top_n=5):
    try:
        cast = ast.literal_eval(cast_str)
        return [c["name"] for c in cast[:top_n]]
    except:
        return []

def get_director(crew_str):
    try:
        crew = ast.literal_eval(crew_str)
        for c in crew:
            if c.get("job") == "Director":
                return c.get("name")
    except:
        pass
    return ""

credits["cast_names"] = credits["cast"].apply(get_cast)
credits["director"] = credits["crew"].apply(get_director)

movies = movies.merge(
    credits[["id", "cast_names", "director"]],
    left_on="movieId",
    right_on="id",
    how="left"
)

movies["cast_names"] = movies["cast_names"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
movies["director"] = movies["director"].fillna("")

movies["combined_features"] = (
    movies["genres"].fillna("") + " " +
    movies["cast_names"] + " " +
    movies["director"]
)

# -----------------------------
# RECOMMENDERS
# -----------------------------
def recommend_movies_content(movie_id, top_n=10):
    idx = movies[movies.movieId == movie_id].index[0]
    sim_scores = sorted(
        enumerate(cosine_sim[idx]),
        key=lambda x: x[1],
        reverse=True
    )[1:top_n+1]
    indices = [i[0] for i in sim_scores]
    return movies.iloc[indices][["movieId", "title"]]

def rank_movies_cf(user_id, movie_ids, top_n=10):
    preds = []
    for mid in movie_ids:
        try:
            preds.append((mid, svd.predict(user_id, mid).est))
        except:
            pass
    return sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]

def generate_rich_explanation(base_movie_id, rec_movie_id):
    base = movies[movies.movieId == base_movie_id].iloc[0]
    rec = movies[movies.movieId == rec_movie_id].iloc[0]

    reasons = []
    if base.director and base.director == rec.director:
        reasons.append(f"Same director: {base.director}")

    common = set(base.genres.split("|")) & set(rec.genres.split("|"))
    if common:
        reasons.append(f"Shared genres: {', '.join(common)}")

    return " • ".join(reasons) if reasons else "Similar content and user preferences"
