import contextlib
import io
import unittest
from unittest.mock import patch

import pandas as pd
import requests

import utils_tools2


class TerminalOutputTests(unittest.TestCase):
    def test_prints_text_and_dataframe(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            utils_tools2.print_html("Hello", title="Message")
            utils_tools2.print_html(pd.DataFrame({"name": ["Alice"]}), title="People")
        self.assertIn("Message:\nHello", output.getvalue())
        self.assertIn("People:", output.getvalue())
        self.assertIn("Alice", output.getvalue())
        self.assertNotIn("IPython", output.getvalue())

    def test_http_response_prints_json_and_returns_parsed_data(self):
        response = requests.Response()
        response.status_code = 201
        response._content = b'{"id": 42}'
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = utils_tools2.pretty_display("POST /send", response)
        self.assertEqual(result, {"id": 42})
        self.assertIn("POST /send: HTTP 201", output.getvalue())
        self.assertIn('"id": 42', output.getvalue())
        self.assertNotIn("IPython", output.getvalue())

    def test_http_response_handles_non_json_text(self):
        response = requests.Response()
        response.status_code = 500
        response._content = b"Server error"
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = utils_tools2.pretty_display("GET /emails", response)
        self.assertEqual(result, "Server error")
        self.assertIn("Server error", output.getvalue())


class EmailServerConfigurationTests(unittest.TestCase):
    def test_missing_or_invalid_url_fails_before_request(self):
        for value in (None, "", "   ", "localhost:5000", "ftp://localhost:5000"):
            with self.subTest(value=value), patch.object(utils_tools2, "BASE_URL", value):
                with patch.object(utils_tools2.session, "post") as post:
                    with self.assertRaisesRegex(ValueError, "M3_EMAIL_SERVER_API_URL"):
                        utils_tools2.test_send_email()
                    post.assert_not_called()

    def test_send_uses_configured_url(self):
        with patch.object(utils_tools2, "BASE_URL", "http://127.0.0.1:5000/"):
            with patch.object(utils_tools2.session, "post") as post, patch.object(utils_tools2, "pretty_display"):
                utils_tools2.test_send_email()
                post.assert_called_once_with(
                    "http://127.0.0.1:5000/send",
                    json={
                        "recipient": "test@example.com",
                        "subject": "Test Subject",
                        "body": "This is a test email body.",
                    },
                )


if __name__ == "__main__":
    unittest.main()
