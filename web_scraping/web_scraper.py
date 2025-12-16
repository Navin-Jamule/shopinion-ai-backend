import time, random, re
import pandas as pd
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# --- CONFIGURE YOUR CHROME PROFILE PATH ---
USER_DATA_DIR = r"C:\Users\Navin\AppData\Local\Google\Chrome\User Data"
PROFILE_DIR = "Default"   # change if your Chrome profile is different

def extract_asin(product_url: str) -> str:
    """Extract ASIN from Amazon product URL"""
    path = urlparse(product_url).path
    m = re.search(r'/dp/([A-Z0-9]{10})', path)
    if m:
        return m.group(1)
    raise ValueError("ASIN not found in URL")

def init_driver():
    """Initialize Chrome with your logged-in profile"""
    options = webdriver.ChromeOptions()
    options.add_argument(f"user-data-dir={USER_DATA_DIR}")
    options.add_argument(f"profile-directory={PROFILE_DIR}")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def normalize_rating(rating_text: str) -> int:
    """Convert '4.0 out of 5 stars' -> 4"""
    m = re.search(r'(\d+(\.\d+)?)', rating_text)
    return int(float(m.group(1))) if m else None

def scrape_reviews(product_url, max_pages=5):
    """Scrape Amazon reviews for a given product URL"""
    asin = extract_asin(product_url)
    base_url = f"https://www.amazon.in/product-reviews/{asin}/"

    driver = init_driver()
    all_data = []

    star_filters = ["five_star", "four_star", "three_star", "two_star", "one_star"]

    for star_key in star_filters:
        page = 1
        while page <= max_pages:
            url = f"{base_url}?filterByStar={star_key}&reviewerType=all_reviews&pageNumber={page}"
            print(f"Scraping: {url}")
            driver.get(url)
            time.sleep(random.uniform(3, 5))

            # If redirected to login, pause for manual login
            if "signin" in driver.current_url.lower():
                print("⚠️ Amazon redirected to login. Please log in manually.")
                input("👉 Press Enter here once logged in...")
                driver.get(url)
                time.sleep(3)

            # Scroll to bottom to load lazy content
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(2, 4))

            soup = BeautifulSoup(driver.page_source, "html.parser")
            review_blocks = soup.select('li[data-hook="review"], div[data-hook="review"]')

            if not review_blocks:
                print("⚠️ No reviews found on this page.")
                break

            for block in review_blocks:
                text = block.select_one('span[data-hook="review-body"] > span')
                rating = block.select_one('i[data-hook="review-star-rating"] > span.a-icon-alt')
                title = block.select_one('a[data-hook="review-title"] > span')
                date = block.select_one('span[data-hook="review-date"]')
                verified = block.select_one('span[data-hook="avp-badge"]')
                format_info = block.select_one('a[data-hook="format-strip"]')
                images = block.select('img[data-hook="review-image-tile"]')
                image_urls = [img.get('src') for img in images if img.get('src')]

                if text and rating:
                    all_data.append({
                        "review": text.get_text(strip=True),
                        "rating": normalize_rating(rating.get_text(strip=True)),
                        "title": title.get_text(strip=True) if title else None,
                        "date": date.get_text(strip=True) if date else None,
                        "verified": verified.get_text(strip=True) if verified else None,
                        "format": format_info.get_text(strip=True) if format_info else None,
                        "images": ", ".join(image_urls) if image_urls else None
                    })

            # Check for "Next" button
            next_btn = soup.select_one("li.a-last a")
            if not next_btn:
                break
            page += 1

    driver.quit()

    # Deduplicate reviews
    df = pd.DataFrame(all_data)
    df.drop_duplicates(subset=["review"], inplace=True)

    return df, asin

# --- RUN SCRIPT ---
if __name__ == "__main__":
    # Example product link
    product_url = "https://www.amazon.in/CMF-Phone-Light-Green-Storage/dp/B0F77VWZM5/ref=sr_1_1?sr=8-1"
    df, asin = scrape_reviews(product_url, max_pages=5)
    print(df.head())
    df.to_csv(f"web_scraping/{asin}_reviews.csv", index=False, encoding="utf-8-sig")
    print(f"✅ Saved {len(df)} unique reviews to web_scraping/{asin}_reviews.csv")