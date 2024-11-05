import requests

response = requests.get("http://127.0.0.1:8000/recommend/1?num_recommendations=5")
print(response.json())
