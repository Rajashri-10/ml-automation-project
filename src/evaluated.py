import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data/processed/processed_data.csv")
model_path = os.path.join(BASE_DIR, "model.pkl")
history_path = os.path.join(BASE_DIR, "metrics/history.csv")

df = pd.read_csv(data_path)
X = df[["feature1", "feature2"]]
y = df["label"]

model = joblib.load(model_path)
preds = model.predict(X)

accuracy = accuracy_score(y, preds)
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

os.makedirs(os.path.dirname(history_path), exist_ok=True)


if os.path.exists(history_path):
    history = pd.read_csv(history_path)
else:
    history = pd.DataFrame(columns=["timestamp", "accuracy"])

history.loc[len(history)] = [timestamp, accuracy]
history.to_csv(history_path, index=False)

print(f"📊 Accuracy logged: {accuracy}")
