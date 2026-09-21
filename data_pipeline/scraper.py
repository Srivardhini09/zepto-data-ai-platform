import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

# Fixed conversion rate required by the assignment
GBP_TO_INR = 105.50

# Convert text ratings into numbers
RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}


def get_soup(url):
    """Download a webpage and return its BeautifulSoup object."""
    
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    return BeautifulSoup(response.content, "html.parser")


def scrape_category(category_name, category_url):
    """Scrape all books from one category."""

    books = []

    current_url = category_url

    while current_url:

        print("Scraping:", current_url)

        soup = get_soup(current_url)

        # Find every book on the current page
        book_cards = soup.select("article.product_pod")

        for book in book_cards:

            # Title
            title = book.h3.a["title"]

            # Price
            price_text = book.select_one(".price_color").get_text(strip=True)
            # Extract numeric part safely, even if the £ symbol is incorrectly encoded
            price_match = re.search(r"\d+(?:\.\d+)?", price_text)
            if price_match:
                price_gbp = float(price_match.group())
            else:
                price_gbp = None



            # Star rating
            rating_element = book.select_one(".star-rating")

            rating_text = rating_element["class"][1]

            rating = RATING_MAP.get(rating_text)

            # Availability
            availability = book.select_one(
                ".availability"
            ).get_text(" ", strip=True)

            in_stock = "In stock" in availability

            books.append({
                "title": title,
                "price_gbp": price_gbp,
                "star_rating": rating_text,
                "rating": rating,
                "availability": availability,
                "in_stock": in_stock,
                "category": category_name
            })

        # Find the next page
        next_button = soup.select_one("li.next a")

        if next_button:

            next_url = next_button["href"]

            current_url = (
                current_url.rsplit("/", 1)[0]
                + "/"
                + next_url
            )

        else:
            current_url = None

        # Small delay between requests
        time.sleep(0.2)

    return books


def main():

    # Three categories required by the assignment
    categories = {
        "Travel":
            "https://books.toscrape.com/catalogue/category/books/travel_2/index.html",

        "Mystery":
            "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html",

        "Historical Fiction":
            "https://books.toscrape.com/catalogue/category/books/historical-fiction_4/index.html"
    }

    all_books = []

    # Scrape every category
    for category_name, category_url in categories.items():

        category_books = scrape_category(
            category_name,
            category_url
        )

        all_books.extend(category_books)

    # Convert list to DataFrame
    df = pd.DataFrame(all_books)

    print("\n--------------------------------")
    print("RAW DATA")
    print("--------------------------------")

    print(df.head())

    print("\nTotal books scraped:", len(df))

    # --------------------------------
    # DATA CLEANING
    # --------------------------------

    # Price: median imputation if parsing failed
    if df["price_gbp"].isna().any():

        median_price = df["price_gbp"].median()

        df["price_gbp"] = df["price_gbp"].fillna(
            median_price
        )

    # Rating: median imputation if parsing failed
    if df["rating"].isna().any():

        median_rating = int(
            df["rating"].median()
        )

        df["rating"] = df["rating"].fillna(
            median_rating
        )

    # Convert GBP to INR using the required fixed rate
    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    )

    # Ensure correct data types
    df["price_gbp"] = df["price_gbp"].astype(float)

    df["price_inr"] = df["price_inr"].astype(float)

    df["rating"] = df["rating"].astype(int)

    df["in_stock"] = df["in_stock"].astype(bool)

    # Save cleaned dataset
    output_file = "data_pipeline/books_cleaned.csv"

    df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------
    # FINAL INFORMATION
    # --------------------------------

    print("\n--------------------------------")
    print("CLEANED DATA")
    print("--------------------------------")

    print(df.head())

    print("\nDataset shape:", df.shape)

    print("\nBooks per category:")
    print(df["category"].value_counts())

    print("\nData types:")
    print(df.dtypes)

    print("\nGBP → INR conversion rate:")
    print("1 GBP =", GBP_TO_INR, "INR")

    print("\nSaved successfully to:")
    print(output_file)


if __name__ == "__main__":
    main()