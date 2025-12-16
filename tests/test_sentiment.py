import pandas as pd
from backend.services.model_service import run_model

def test_run_model_output_structure():
    df = pd.DataFrame({
        "review": ["Great display", "Battery drains fast"],
        "rating": [5, 2]  # Add this line
    })
    asin = "TEST123"
    result = run_model(df, asin)

    assert result["asin"] == asin
    assert result["total_reviews"] == 2
    assert isinstance(result["summary"], list)
    assert isinstance(result["reviews"], list)
    assert all("predicted_sentiment" in r for r in result["reviews"])