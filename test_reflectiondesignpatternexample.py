import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import reflectiondesignpatternexample
import utils


class StandaloneWorkflowTests(unittest.TestCase):
    def test_workflow_creates_both_charts_and_reflects_on_executed_code(self):
        with tempfile.TemporaryDirectory() as directory:
            base = str(Path(directory) / "chart")
            df = pd.DataFrame({"price": [1, 2]})
            def chart_code(path):
                return (
                    "<execute_python>\nimport matplotlib.pyplot as plt\n"
                    f"plt.plot([1, 2])\nplt.savefig({path!r})\nplt.close()\n</execute_python>"
                )

            v1 = chart_code(base + "_v1.png")
            v2 = chart_code(base + "_v2.png")
            with (
                patch("utils.load_and_prepare_data", return_value=df),
                patch("reflectiondesignpatternexample.generate_chart_code", return_value=v1),
                patch("utils.image_openai_call", return_value='{"feedback": "Looks good"}\n' + v2) as reflect,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = reflectiondesignpatternexample.run_workflow("input.csv", "make a chart", "gpt-4o-mini", "gpt-5-mini", base)

            self.assertTrue(Path(result["chart_v1"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue(Path(result["chart_v2"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(result["feedback"], "Looks good")
            self.assertIn(utils.extract_code(v1), reflect.call_args.args[1])

    def test_missing_code_or_chart_stops_workflow(self):
        df = pd.DataFrame()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "chart.png")
            with self.assertRaisesRegex(ValueError, "did not contain"):
                reflectiondesignpatternexample.execute_chart("no code", df, "model", path)
            with self.assertRaisesRegex(FileNotFoundError, "did not create"):
                reflectiondesignpatternexample.execute_chart("<execute_python>pass</execute_python>", df, "model", path)
            Path(path).write_bytes(b"old")
            with self.assertRaisesRegex(RuntimeError, "did not update"):
                reflectiondesignpatternexample.execute_chart("<execute_python>pass</execute_python>", df, "model", path)

    def test_invalid_reflection_is_reported(self):
        with patch("utils.encode_image_b64", return_value=("image/png", "data")):
            with patch("utils.image_openai_call", return_value='{"feedback": "ok"}'):
                with self.assertRaisesRegex(ValueError, "did not contain executable code"):
                    reflectiondesignpatternexample.reflect_on_image_and_regenerate("chart.png", "chart", "model", "v2.png", "code")

    def test_generated_code_failure_is_raised_after_retry(self):
        with patch("utils.get_response", return_value="<execute_python>raise ValueError('again')</execute_python>"):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "again"):
                    utils.run_generated_code("raise ValueError('first')", pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
