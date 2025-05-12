from dotenv import load_dotenv
import os
from kfp import compiler
from google.cloud import aiplatform

from pipelines.predict_price import uber_price_pipeline

print("Loading environment variables...")
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
BUCKET_NAME = os.getenv("BUCKET_NAME")
PIPELINE_BUCKET_NAME = os.getenv("PIPELINE_BUCKET_NAME")

print("\nCompiling pipeline...")
compiler.Compiler().compile(
    pipeline_func=uber_price_pipeline,
    package_path="uber_pipeline.json"
)
print("Pipeline compilation completed. Output saved to uber_pipeline.json")

print("\nInitializing Vertex AI...")
aiplatform.init(project=PROJECT_ID, location="us-central1")

print("\nCreating and running pipeline job...")
job = aiplatform.PipelineJob(
    display_name="uber-price-pipeline",
    template_path="uber_pipeline.json",
    pipeline_root=f"gs://{PIPELINE_BUCKET_NAME}/pipeline-root/"
)
print("Pipeline job created. Starting execution...")
job.run()
print("Pipeline job submitted successfully!")
