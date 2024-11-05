import pandas as pd


ratings = pd.read_csv('ratings.csv')


user_ids = ratings['userId'].unique()
print("Available user IDs:", user_ids)
