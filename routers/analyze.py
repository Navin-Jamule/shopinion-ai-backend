import os
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services.model_service import run_model
from backend.web_scraping.web_scraper import scrape_reviews

router = APIRouter()

class ProductRequest(BaseModel):
    url: str

@router.post("/analyze")
async def analyze_product(req: ProductRequest):
    # Step 1: Scrape reviews
    df, asin = scrape_reviews(req.url)

    # Step 2: Run model
    result = run_model(df, asin)

    # Step 3: Save output JSON
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{asin}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result

@router.get("/results/{asin}")
async def get_results(asin: str):
    path = os.path.join("outputs", f"{asin}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Result not found")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)