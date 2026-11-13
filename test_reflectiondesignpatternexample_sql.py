import contextlib
import io
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import reflectiondesignpatternexample_sql as sql_example
import utils_sql


class SQLWorkflowTests(unittest.TestCase):
    def test_main_generates_and_refines_query_without_changing_database(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "products.db"
            with contextlib.closing(sqlite3.connect(path)) as conn, conn:
                conn.execute("CREATE TABLE transactions (color TEXT, qty_delta INTEGER)")
                conn.executemany("INSERT INTO transactions VALUES (?, ?)", [("red", -4), ("blue", -2)])

            first = "SELECT color FROM transactions"
            second = "SELECT color, -SUM(qty_delta) AS sales FROM transactions GROUP BY color ORDER BY sales DESC LIMIT 1"
            response = '{"feedback": "Rank colors by sales.", "refined_sql": "' + second + '"}'
            output = io.StringIO()
            with (
                patch.object(sys, "argv", ["program", "--database", str(path)]),
                patch("utils.get_response", side_effect=[first, response]) as model,
                contextlib.redirect_stdout(output),
            ):
                sql_example.main()

            self.assertIn("Rank colors by sales.", output.getvalue())
            self.assertIn("red", output.getvalue())
            self.assertIn("blue", model.call_args_list[1].args[1])
            with contextlib.closing(sqlite3.connect(path)) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0], 2)

    def test_queries_are_read_only_and_errors_are_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "products.db"
            with contextlib.closing(sqlite3.connect(path)) as conn, conn:
                conn.execute("CREATE TABLE transactions (color TEXT)")
                conn.execute("INSERT INTO transactions VALUES ('red')")

            for query in ("DELETE FROM transactions", "WITH x AS (SELECT 1) DELETE FROM transactions"):
                with self.subTest(query=query), self.assertRaises((ValueError, pd.errors.DatabaseError)):
                    utils_sql.execute_sql(query, str(path))
            with self.assertRaises(pd.errors.DatabaseError):
                utils_sql.execute_sql("SELECT missing FROM transactions", str(path))
            with contextlib.closing(sqlite3.connect(path)) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0], 1)

            missing = Path(directory) / "missing.db"
            with self.assertRaises(sqlite3.OperationalError):
                utils_sql.get_schema(str(missing))
            self.assertFalse(missing.exists())

    def test_sample_creation_never_replaces_existing_database(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "sample.db")
            with contextlib.redirect_stdout(io.StringIO()):
                utils_sql.create_transactions_db(path, n_products=1, n_txns_per_product=1)
            with self.assertRaises(FileExistsError):
                utils_sql.create_transactions_db(path, n_products=1, n_txns_per_product=1)
            with contextlib.closing(sqlite3.connect(path)) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0], 1)

    def test_invalid_refinement_does_not_silently_reuse_original_query(self):
        for content in ("not json", "[]", '{"feedback": "ok"}', '{"feedback": "ok", "refined_sql": ""}'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                sql_example.parse_refinement(content)

    def test_refinement_accepts_json_fences(self):
        content = '```json\n{"feedback": "ok", "refined_sql": "SELECT 1"}\n```'
        self.assertEqual(sql_example.parse_refinement(content), ("ok", "SELECT 1"))


if __name__ == "__main__":
    unittest.main()
