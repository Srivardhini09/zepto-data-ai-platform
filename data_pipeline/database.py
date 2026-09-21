import sqlite3
import pandas as pd

CSV_PATH = "data_pipeline/books_cleaned.csv"
DB_PATH = "data_pipeline/books.db"


def create_database():
    # Read cleaned CSV
    df = pd.read_csv(CSV_PATH)

    # Connect to SQLite
    conn = sqlite3.connect(DB_PATH)

    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")

    cursor = conn.cursor()

    # Create categories table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        )
    """)

    # Create books table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL,
            star_rating TEXT,
            rating INTEGER,
            availability TEXT,
            in_stock INTEGER,
            price_inr REAL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    # Insert categories
    for category in df["category"].unique():
        cursor.execute(
            "INSERT OR IGNORE INTO categories (category_name) VALUES (?)",
            (category,)
        )

    # Insert books
    for _, row in df.iterrows():

        cursor.execute(
            "SELECT category_id FROM categories WHERE category_name = ?",
            (row["category"],)
        )

        category_id = cursor.fetchone()[0]

        cursor.execute("""
            INSERT INTO books (
                title,
                price_gbp,
                star_rating,
                rating,
                availability,
                in_stock,
                price_inr,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["title"],
            row["price_gbp"],
            row["star_rating"],
            row["rating"],
            row["availability"],
            int(row["in_stock"]),
            row["price_inr"],
            category_id
        ))

    conn.commit()

    # Display table counts
    category_count = cursor.execute(
        "SELECT COUNT(*) FROM categories"
    ).fetchone()[0]

    book_count = cursor.execute(
        "SELECT COUNT(*) FROM books"
    ).fetchone()[0]

    print("Database created successfully!")
    print("Categories:", category_count)
    print("Books:", book_count)
    print("Database saved to:", DB_PATH)

    conn.close()


if __name__ == "__main__":
    create_database()