import os
import pandas as pd
from backend.model.model import run_absa_pipeline

def run_model(df: pd.DataFrame, asin: str):
    #Ensure outputs/ folder exists
    os.makedirs("outputs", exist_ok=True)

    #Save raw reviews
    csv_path = f"outputs/{asin}_raw.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    #Run ABSA pipeline
    aspect_csv, summary_csv = run_absa_pipeline(csv_path, asin)

    #Load results
    df_summary = pd.read_csv(summary_csv)
    df_aspects = pd.read_csv(aspect_csv)

    #Return structured response
    return {
        "asin": asin,
        "total_reviews": len(df),
        "summary": df_summary.to_dict(orient="records"),
        "reviews": df_aspects[["review", "aspect", "predicted_sentiment", "confidence"]].to_dict(orient="records")
    }