# 🚕 Uber SF Dynamic Price Prediction (Vertex AI + KFP)

This project builds an end-to-end ML pipeline to predict ride prices in San Francisco based on traversal count, location, and day type. It uses Google Cloud's Vertex AI Pipelines, BigQuery, Keras, and Kubeflow Pipelines (KFP) for scalable training and deployment.

---

## 📦 Features

- Ingest Uber SF ride data into BigQuery
- Preprocess with one-hot encoding + scaling
- Train Keras model in chunks using TensorFlow
- Deploy model to Vertex AI endpoint
- Predict prices from local input script

---

## 🧠 Tech Stack

- Google Cloud Platform
  - Vertex AI Pipelines (KFP v2)
  - Vertex AI Models & Endpoints
  - BigQuery
  - Cloud Storage
- Python 3.12
- TensorFlow / Keras
- pandas, numpy, scikit-learn
- dotenv

---

## 🔧 Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create a `.env` file

In the root directory, create a file named `.env` and add:

    PROJECT_ID=
    ENDPOINT_ID=
    LOCATION=

**Note:** Do not commit this file to version control.

---

## 💡 Pipeline Overview

1. Upload dataset to a GCS bucket.
2. Load it into BigQuery (`uber_sf.ride_prices`).
3. Run the Vertex AI pipeline:
   - Extracts and preprocesses data
   - Applies one-hot encoding and scaling
   - Trains a neural network model in chunks
   - Saves model + metadata to GCS
4. Deploy model to Vertex AI endpoint.

---

## 🚀 Predict Locally

After deployment, run:

    python scripts/predict.py

Make sure these files are available in your repo root:

- `expected_columns.txt`
- `.env`

---

## 📂 Project Structure

    uber_sf/
    ├── scripts/
    │   └── predict.py               # Prediction client
    ├── expected_columns.txt         # Exported column names
    ├── processed.csv                # Optional local preview
    ├── train_model_component.py     # KFP training component
    ├── requirements.txt
    └── README.md

---

## ⚠️ Known Limitation

This model uses:

    keras.mixed_precision.set_global_policy('mixed_float16')

But Vertex AI’s prediction service **does not support float16 (half)** in JSON.

**To resolve:**  
Remove or comment out the mixed precision line and retrain/export the model with `float32`.

---

## 👤 Author

Built by Pranav using Google Cloud tools to demonstrate real-world ML system deployment.
