from kfp import dsl
from components.extract_and_process import extract_and_process
from components.train_model import train_model

@dsl.pipeline(name="uber-price-prediction-pipeline")
def uber_price_pipeline():
    processed = extract_and_process()
    
    trained = train_model(input_data=processed.output)
    trained.set_cpu_limit("16")
    trained.set_memory_limit("128G") 
