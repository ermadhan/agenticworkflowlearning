import contextlib
import io
import runpy
import unittest
from unittest.mock import patch
from pathlib import Path

import utils


class AICreditClientTests(unittest.TestCase):
    def test_second_example_initializes_configured_client(self):
        values = {"AI_CREDIT_KEY": "test-key", "AI_CREDIT_URL": "https://gateway.example/v1"}
        path = Path(__file__).with_name("tools_example2.py")
        with (
            patch.dict("os.environ", values, clear=True),
            patch("utils.OpenAI") as client,
            patch("utils_tools2.test_send_email", return_value={"id": 1}),
            patch("utils_tools2.test_get_email"),
            patch("email_tools.send_email", return_value={"id": 1}),
            patch("email_tools.get_email", return_value={"id": 1}),
            patch("utils_tools2.reset_database"),
            patch("utils_tools2.run_email_agent", return_value="Done"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            namespace = runpy.run_path(str(path))
            client.assert_called_once_with(api_key="test-key", base_url="https://gateway.example/v1")
            self.assertIs(namespace["client"], client.return_value)

    def test_requires_api_key_and_url(self):
        for values, missing in (
            ({}, "AI_CREDIT_KEY"),
            ({"AI_CREDIT_KEY": "test-key"}, "AI_CREDIT_URL"),
            ({"AI_CREDIT_KEY": "test-key", "AI_CREDIT_URL": "  "}, "AI_CREDIT_URL"),
        ):
            with self.subTest(missing=missing, values=values):
                with patch.dict("os.environ", values, clear=True):
                    with patch("utils.OpenAI") as client:
                        with self.assertRaisesRegex(ValueError, missing):
                            utils.get_response("gpt-4o-mini", "Hello")
                        client.assert_not_called()

    def test_chat_completion_uses_configured_gateway(self):
        values = {
            "AI_CREDIT_KEY": "test-key",
            "AI_CREDIT_URL": "https://gateway.example/v1",
        }
        with patch.dict("os.environ", values, clear=True), patch("utils.OpenAI") as client:
            completion = client.return_value.chat.completions.create
            completion.return_value.choices[0].message.content = "Generated code"

            self.assertEqual(utils.get_response("gpt-4o-mini", "Hello"), "Generated code")

            client.assert_called_once_with(
                api_key="test-key", base_url="https://gateway.example/v1"
            )
            completion.assert_called_once_with(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
            )

    def test_missing_text_is_reported(self):
        values = {"AI_CREDIT_KEY": "test-key", "AI_CREDIT_URL": "https://gateway.example/v1"}
        with patch.dict("os.environ", values, clear=True), patch("utils.OpenAI") as client:
            client.return_value.chat.completions.create.return_value.choices[0].message.content = None
            with self.assertRaisesRegex(ValueError, "without text content"):
                utils.get_response("gpt-4o-mini", "Hello")

    def test_image_request_uses_chat_completions(self):
        values = {"AI_CREDIT_KEY": "test-key", "AI_CREDIT_URL": "https://gateway.example/v1"}
        with patch.dict("os.environ", values, clear=True), patch("utils.OpenAI") as client:
            completion = client.return_value.chat.completions.create
            completion.return_value.choices[0].message.content = "feedback"
            self.assertEqual(utils.image_openai_call("model", "prompt", "image/png", "YWJj"), "feedback")
            completion.assert_called_once_with(
                model="model",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "prompt"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj"}},
                    ],
                }],
            )


if __name__ == "__main__":
    unittest.main()
