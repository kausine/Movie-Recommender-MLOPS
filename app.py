from fastapi import FastAPI, Query
import joblib
import pandas as pd
import kagglehub

# Load trained model
model = joblib.load("svd_model.joblib")

# Load dataset
path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")
ratings = pd.read_csv(f"{path}/ratings_small.csv")
movies = pd.read_csv(f"{path}/movies_metadata.csv")

# Clean movie IDs
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies = movies.dropna(subset=['id']).drop_duplicates(subset=['id'])
id_to_title = movies.set_index('id')['title'].to_dict()

# Create FastAPI app
app = FastAPI(title="Movie Recommender API")

def recommend_top_n(user_id, model, movies_df, n=5):
    movie_ids = pd.to_numeric(movies_df['id'], errors='coerce').dropna().astype(int).unique()
    user_rated = ratings[ratings['userId'] == user_id]['movieId'].values
    predictions = []
    for mid in movie_ids:
        if mid not in user_rated:
            pred = model.predict(user_id, mid)
            predictions.append((mid, pred.est))
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]
    return [(mid, id_to_title.get(mid, "Unknown"), round(score, 2)) for mid, score in top_n]

@app.get("/recommend")
def recommend(user_id: int = Query(..., description="User ID"),
              n: int = Query(5, description="Number of recommendations")):
    recs = recommend_top_n(user_id, model, movies, n)
    return {"user_id": user_id, "recommendations": recs}

# 👇 Add this new route
@app.get("/")
def read_root():
    return {"message": "Movie Recommender API is live! 🚀 Use /recommend?user_id=1&n=5"}
