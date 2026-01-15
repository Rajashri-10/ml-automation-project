import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

raw_path = os.path.join(BASE_DIR, "data/raw/raw_data.csv")
processed_path = os.path.join(BASE_DIR, "data/processed/processed_data.csv")

df = pd.read_csv(raw_path)

# Simple preprocessing (example)
df = df.dropna()

os.makedirs(os.path.dirname(processed_path), exist_ok=True)
df.to_csv(processed_path, index=False)

print(" Data preprocessing completed")
