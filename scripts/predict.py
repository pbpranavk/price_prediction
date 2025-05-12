import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google.cloud import aiplatform

# --- Load environment variables ---
print("🔧 Loading environment variables...")
load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
ENDPOINT_ID = os.getenv("ENDPOINT_ID")
LOCATION = os.getenv("LOCATION", "us-central1")

if not all([PROJECT_ID, ENDPOINT_ID]):
    raise ValueError("❌ Set PROJECT_ID and ENDPOINT_ID in your .env file")

print(f"✅ Using Project ID: {PROJECT_ID}")
print(f"✅ Using Endpoint ID: {ENDPOINT_ID}")
print(f"✅ Using Location: {LOCATION}")

# --- Load expected columns from training ---
print("\n📄 Loading expected columns...")
with open("expected_columns.txt") as f:
    EXPECTED_COLS = f.read().splitlines()
print(f"✅ Loaded {len(EXPECTED_COLS)} expected columns")

# --- Hardcoded scaler parameters ---
NUMERICAL_COLS = ['traversals', 'lat', 'lng']
MEAN = np.array([66.65438, 37.77983617, -122.42894641])
SCALE = np.array([52.5958995, 0.0175805019, 0.0276455556])

# --- Raw input ---
print("\n🧾 Preparing input...")
raw_input = pd.DataFrame([{
    "hexid": "8928308280fffff",
    "dayType": "weekday",
    "traversals": 66.0,
    "lat": 37.7749,
    "lng": -122.4194
}])
print(raw_input)

# --- One-hot encode categorical features ---
print("\n🔢 One-hot encoding categorical features...")
X = pd.get_dummies(raw_input, columns=["hexid", "dayType"])

# --- Align with training-time expected columns ---
print("📐 Aligning with expected columns...")
X = X.reindex(columns=EXPECTED_COLS, fill_value=0)

# --- Scale numerical features ---
print("\n📏 Scaling numerical features...")
for col, mean, scale in zip(NUMERICAL_COLS, MEAN, SCALE):
    if col in X.columns:
        X[col] = (X[col] - mean) / scale
        print(f"✅ Scaled '{col}': mean={mean}, scale={scale}")
    else:
        print(f"⚠️  '{col}' missing — skipped scaling")

# --- Ensure float32 for compatibility ---
X = X.astype(np.float32)

# --- Predict using Vertex AI ---
print("\n🚀 Making prediction request to Vertex AI...")
aiplatform.init(project=PROJECT_ID, location=LOCATION)
endpoint = aiplatform.Endpoint(ENDPOINT_ID)
instances = [[float(val) for val in row] for row in X.to_numpy()]
prediction = endpoint.predict(instances=instances)

# --- Output result ---
print("\n✅ Prediction Result:")
print("Input shape:", X.shape)
print("Predicted price:", prediction.predictions[0])
