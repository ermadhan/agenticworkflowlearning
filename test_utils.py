import unittest
from unittest.mock import patch

import utils


class AICreditClientTests(unittest.TestCase):
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
