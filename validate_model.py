from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import joblib

data = load_iris()

model = joblib.load("model.pkl")

predictions = model.predict(data.data)

accuracy = accuracy_score(data.target, predictions)

print("Model Accuracy:", accuracy)

assert accuracy >= 0.90, "Model validation failed!"

print("Model validation successful!")
