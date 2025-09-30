# 🎬 Movie Recommender API

This project is a **FastAPI-based Movie Recommendation System** built using:
- Matrix Factorization (SVD)
- FastAPI for serving predictions
- Kaggle Movies Dataset

## 🚀 How it works
- User sends a request like:
/recommend?user_id=1&n=5

- API returns top N movie recommendations for that user.

## 🛠️ Tech stack
- FastAPI
- Uvicorn
- Pandas / NumPy
- Surprise (SVD model)
- KaggleHub

## 🌍 Deployment
Will be deployed on **Render** with a public API endpoint.
