import requests

url = "http://localhost:9696/predict"

# client = {"job": "student", "duration": 280, "poutcome": "failure"}

client = {"job": "management", "duration": 400, "poutcome": "success"}
response = requests.post(url, json=client).json()

print(f"client: {client}, response: {response}")
