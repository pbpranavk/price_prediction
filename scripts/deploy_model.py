from dotenv import load_dotenv
import os

from google.cloud import aiplatform

load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME")

aiplatform.init(project=PROJECT_ID, location="us-central1")

model = aiplatform.Model.upload(
    display_name="uber-price-model",
    artifact_uri=f"gs://{MODEL_BUCKET_NAME}",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest"
)

endpoint = model.deploy(machine_type="n1-standard-2")

print(f"Model deployed to endpoint: {endpoint.name}")
