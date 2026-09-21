import sqlite3
import pandas as pd

DB_PATH = "data_pipeline/books.db"


def run_query(conn, query, title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    result = pd.read_sql(query, conn)
    print(result.to_string(index=False))

    return result


def main():
    conn = sqlite3.connect(DB_PATH)

    # 1. SELECT + WHERE
    query1 = """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating >= 4
    """
    result1 = run_query(
        conn,
        query1,
        "QUERY 1: Books with rating >= 4"
    )

    # 2. ORDER BY
    query2 = """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
    """
    result2 = run_query(
        conn,
        query2,
        "QUERY 2: Books ordered by price"
    )

    # 3. LIMIT
    query3 = """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 5
    """
    result3 = run_query(
        conn,
        query3,
        "QUERY 3: Five most expensive books"
    )

    # 4. DISTINCT
    query4 = """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating
    """
    result4 = run_query(
        conn,
        query4,
        "QUERY 4: Distinct ratings"
    )

    # 5. BETWEEN
    query5 = """
        SELECT title, price_gbp, rating
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp
    """
    result5 = run_query(
        conn,
        query5,
        "QUERY 5: Books priced between £20 and £40"
    )

    # 6. JOIN
    query6 = """
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
    result6 = run_query(
        conn,
        query6,
        "QUERY 6: Books with category using JOIN"
    )

    # Save SQL queries and outputs
    with open("data_pipeline/query_results.txt", "w", encoding="utf-8") as f:

        queries = [
            ("QUERY 1", query1, result1),
            ("QUERY 2", query2, result2),
            ("QUERY 3", query3, result3),
            ("QUERY 4", query4, result4),
            ("QUERY 5", query5, result5),
            ("QUERY 6", query6, result6),
        ]

        for name, query, result in queries:
            f.write("\n" + "=" * 60 + "\n")
            f.write(name + "\n")
            f.write("=" * 60 + "\n")
            f.write("SQL:\n")
            f.write(query.strip() + "\n\n")
            f.write("OUTPUT:\n")
            f.write(result.to_string(index=False))
            f.write("\n")

    conn.close()

    print("\nAll queries executed successfully!")
    print("Query results saved to: data_pipeline/query_results.txt")


if __name__ == "__main__":
    main()