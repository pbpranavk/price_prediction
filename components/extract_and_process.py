from dotenv import load_dotenv
import os

from kfp.dsl import component, Output, Dataset

load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")


@component(
    packages_to_install=["google-cloud-bigquery", "pandas", "shapely", "pyproj", "pyarrow", "db-dtypes"],
    base_image="python:3.12"
)
def extract_and_process(output_data: Output[Dataset]):
    import os
    import pandas as pd
    from shapely import wkt
    from shapely.geometry import Polygon

    from google.cloud import bigquery
    bq = bigquery.Client()

    query = """
    SELECT 
        hexid,
        dayType,
        traversals,
        wktGeometry
    FROM `.uber_sf.ride_prices`
    WHERE traversals IS NOT NULL
    """
    
    df = bq.query(query).to_dataframe()

    # Convert WKT to centroid lat/lng
    def extract_centroid_lat_lng(wkt_str):
        try:
            polygon = wkt.loads(wkt_str)
            centroid = polygon.centroid
            return pd.Series([centroid.y, centroid.x])
        except:
            return pd.Series([None, None])

    df[["lat", "lng"]] = df["wktGeometry"].apply(extract_centroid_lat_lng)
    df.drop(columns=["wktGeometry"], inplace=True)

    # Optionally simulate price as a label
    df["price"] = df["traversals"] * 0.8 + 2  # simple dummy label

    output_path = output_data.path + "/processed.csv"
    os.makedirs(output_data.path, exist_ok=True)
    df.to_csv(output_path, index=False)
