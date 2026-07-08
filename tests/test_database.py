import sqlite3

import pandas as pd
import pytest

from src.database import get_connection, load_dataframe, row_count, run_query, table_schema


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr("src.database.DB_PATH", db_file)
    return db_file


def test_load_dataframe_creates_table(temp_db, sample_raw_df):
    load_dataframe(sample_raw_df, table_name="students")
    result = run_query("SELECT * FROM students;")
    assert len(result) == len(sample_raw_df)
    assert set(result.columns) == set(sample_raw_df.columns)


def test_load_dataframe_replace_overwrites(temp_db, sample_raw_df):
    load_dataframe(sample_raw_df, table_name="students", if_exists="replace")
    load_dataframe(sample_raw_df.iloc[:1], table_name="students", if_exists="replace")
    result = run_query("SELECT * FROM students;")
    assert len(result) == 1


def test_run_query_returns_dataframe(temp_db, sample_raw_df):
    load_dataframe(sample_raw_df, table_name="students")
    result = run_query("SELECT age FROM students WHERE age > ?;", params=(21,))
    assert isinstance(result, pd.DataFrame)
    assert (result["age"] > 21).all()


def test_table_schema_lists_columns(temp_db, sample_raw_df):
    load_dataframe(sample_raw_df, table_name="students")
    schema = table_schema(table_name="students")
    assert "name" in schema.columns
    assert set(schema["name"]) == set(sample_raw_df.columns)


def test_row_count_matches_loaded_data(temp_db, sample_raw_df):
    load_dataframe(sample_raw_df, table_name="students")
    assert row_count(table_name="students") == len(sample_raw_df)


def test_get_connection_closes_after_use(temp_db):
    with get_connection() as conn:
        assert isinstance(conn, sqlite3.Connection)
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1;")

