import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np


ratings = pd.read_csv('ratings.csv')  
movies = pd.read_csv('movies.csv')    


user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)


knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_movie_matrix)


import pickle
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)
user_movie_matrix.to_pickle('user_movie_matrix.pkl')
movies.to_pickle('movies.pkl')

print("Model and data saved.")
