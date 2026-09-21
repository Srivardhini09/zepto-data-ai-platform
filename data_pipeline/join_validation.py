import sqlite3
import pandas as pd

DB_PATH = "data_pipeline/books.db"


def main():
    conn = sqlite3.connect(DB_PATH)

    # -----------------------------
    # 1. SQL JOIN using pd.read_sql
    # -----------------------------
    sql_join = """
        SELECT
            b.title,
            b.price_gbp,
            b.rating,
            c.category_name
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        ORDER BY c.category_name, b.title
    """

    sql_result = pd.read_sql(sql_join, conn)

    print("\nSQL JOIN RESULT:")
    print(sql_result.head(10).to_string(index=False))

    # -----------------------------------
    # 2. Read tables into DataFrames
    # -----------------------------------
    books_df = pd.read_sql(
        "SELECT * FROM books",
        conn
    )

    categories_df = pd.read_sql(
        "SELECT * FROM categories",
        conn
    )

    # -----------------------------------
    # 3. Reproduce JOIN using pd.merge()
    # -----------------------------------
    merge_result = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    merge_result = merge_result[
        ["title", "price_gbp", "rating", "category_name"]
    ].sort_values(
        ["category_name", "title"]
    ).reset_index(drop=True)

    sql_result = sql_result.reset_index(drop=True)

    print("\nPANDAS MERGE RESULT:")
    print(merge_result.head(10).to_string(index=False))

    # -----------------------------------
    # 4. Check equivalence
    # -----------------------------------
    equivalent = sql_result.equals(merge_result)

    print("\n" + "=" * 60)
    print("JOIN EQUIVALENCE CHECK")
    print("=" * 60)

    print("SQL JOIN rows:", len(sql_result))
    print("Pandas merge rows:", len(merge_result))
    print("Results equivalent:", equivalent)

    if equivalent:
        print("SUCCESS: SQL JOIN and pandas.merge() produce the same result.")
    else:
        print("WARNING: Results are not identical.")

    conn.close()


if __name__ == "__main__":
    main()