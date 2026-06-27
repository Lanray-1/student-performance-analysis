import sqlite3
from contextlib import contextmanager

import pandas as pd

from src.config import DB_PATH, TABLE_NAME


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def load_dataframe(df: pd.DataFrame, table_name: str = TABLE_NAME, if_exists: str = "replace") -> None:
    with get_connection() as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)


def run_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def table_schema(table_name: str = TABLE_NAME) -> pd.DataFrame:
    return run_query(f"PRAGMA table_info({table_name});")


def row_count(table_name: str = TABLE_NAME) -> int:
    result = run_query(f"SELECT COUNT(*) AS n FROM {table_name};")
    return int(result["n"].iloc[0])
