import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data/processed/processed_data.csv")
model_path = os.path.join(BASE_DIR, "model.pkl")

df = pd.read_csv(data_path)

X = df[["feature1", "feature2"]]
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, model_path)
print(" Model retrained successfully")
