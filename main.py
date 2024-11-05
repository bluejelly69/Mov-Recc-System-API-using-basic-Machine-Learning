from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle
import numpy as np

app = FastAPI()

# load
with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
user_movie_matrix = pd.read_pickle('user_movie_matrix.pkl')
movies = pd.read_pickle('movies.pkl')

# reccs
def get_movie_recommendations(user_id, num_recommendations=5):
    if user_id not in user_movie_matrix.index:
        raise ValueError("User ID not found in the dataset.")
    
    user_ratings = user_movie_matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = knn_model.kneighbors(user_ratings, n_neighbors=num_recommendations + 1)
    recommendations = []

    for idx in indices.flatten()[1:]:
        movie_ids = user_movie_matrix.columns[user_movie_matrix.iloc[idx] > 0]
        for movie_id in movie_ids:
            movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
            recommendations.append(movie_title)

    return recommendations[:num_recommendations]

# Define the API endpoint
@app.get("/recommend/{user_id}")
def recommend(user_id: int, num_recommendations: int = 5):
    try:
        recommendations = get_movie_recommendations(user_id, num_recommendations)
        return {"user_id": user_id, "recommendations": recommendations}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
