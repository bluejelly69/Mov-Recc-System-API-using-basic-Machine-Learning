# Movie Recommendation System API

This repository implements a **movie recommendation system** powered by **Machine Learning** techniques, specifically **Collaborative Filtering** with **K-Nearest Neighbors (KNN)**. Built using **Python** and deployed as an **API using FastAPI**, this project leverages machine learning to suggest movies to users based on their historical ratings and the preferences of similar users.

## Machine Learning Approach

This project uses **Collaborative Filtering**, a popular machine learning technique for recommendation systems, to make personalized recommendations. The **K-Nearest Neighbors (KNN)** algorithm is used to find users with similar movie preferences and suggest movies that those users have liked. The recommendation model is trained on user rating data from the [MovieLens dataset](https://grouplens.org/datasets/movielens/), where it learns patterns in user behavior to make accurate predictions.

### Key Machine Learning Components

- **Data Preprocessing**: The dataset is transformed into a **user-item matrix** for efficient lookup and comparison, filling in missing values with zero (indicating no rating).
- **Model Training**: Using **KNN with cosine similarity**, the model identifies the nearest neighbors for a given user, focusing on those with similar movie preferences.
- **Model Persistence**: The trained model is saved as a `.pkl` file for reusability, allowing it to be loaded by the FastAPI app without retraining.

## Features

- **Collaborative Filtering**: Recommends movies based on user similarity using K-Nearest Neighbors.
- **FastAPI Deployment**: Exposes a `/recommend/{user_id}` endpoint to request recommendations via an API.
- **Customizable Recommendations**: Allows specifying the number of recommendations to return.
- **Cloud-Ready**: Easily deployable to cloud services such as AWS, Google Cloud, or any platform supporting Python APIs.

## Dataset

The recommendation model is trained using the [MovieLens dataset (small version)](https://grouplens.org/datasets/movielens/). Download the following files and place them in the root directory of this repository:

- `ratings.csv`: Contains user ratings for various movies.
- `movies.csv`: Contains information about the movies, such as IDs and titles.

## Project Structure
project-folder/
│
├── train_model.py                # Script to preprocess data and train the recommendation model
├── main.py                       # FastAPI application for serving recommendations
├── ratings.csv                   # Ratings dataset (download from MovieLens)
├── movies.csv                    # Movies dataset (download from MovieLens)
├── knn_model.pkl                 # Saved KNN model (generated by train_model.py)
├── user_movie_matrix.pkl         # Saved user-item matrix (generated by train_model.py)
└── movies.pkl                    # Saved movies DataFrame (generated by train_model.py)


Usage
Step 1: Train the Model
Run the train_model.py script to preprocess the data, train the model, and save the trained files.

This will save:

knn_model.pkl: The trained KNN model.
user_movie_matrix.pkl: The matrix of user ratings for movies.
movies.pkl: The list of movies and their IDs.

Step 2: Start the FastAPI Server
Launch the API server using uvicorn: uvicorn main:app --reload

The server will start on http://127.0.0.1:8000. You can access the API documentation at http://127.0.0.1:8000/docs.

Step 3: Get Recommendations
Use the /recommend/{user_id} endpoint to get movie recommendations for a specific user. Replace {user_id} with an actual user ID from the dataset and optionally specify the number of recommendations (num_recommendations).

This will return a JSON response with movie recommendations for user_id 1.


Cloud Deployment
The application can be deployed on cloud platforms like AWS or Google Cloud. Use Docker for containerization or deploy directly using services like Elastic Beanstalk, App Engine, or Cloud Run for easy scalability and management.

## License
This project is open-source and available under the MIT License.

