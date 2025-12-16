from backend.web_scraping.web_scraper import scrape_reviews

def run_scraper(url: str):
    df, asin = scrape_reviews(url, max_pages=5)
    return df, asin
