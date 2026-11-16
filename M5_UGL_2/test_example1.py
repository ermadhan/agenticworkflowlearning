import base64
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

import example1


def response(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


def tool_call(name, arguments, call_id):
    return SimpleNamespace(type="function", id=call_id, function=SimpleNamespace(
        name=name, arguments=json.dumps(arguments)
    ))


class StandaloneCampaignTests(unittest.TestCase):
    def test_client_uses_ai_credit_settings(self):
        with (
            patch.object(example1, "load_dotenv"),
            patch.dict("os.environ", {"AI_CREDIT_KEY": " key ", "AI_CREDIT_URL": " https://example.test/v1 "}, clear=True),
            patch.object(example1, "OpenAI") as client,
        ):
            example1.create_client()
            client.assert_called_once_with(api_key="key", base_url="https://example.test/v1")

    def test_full_pipeline_processes_tools_and_writes_outputs(self):
        buffer = io.BytesIO()
        Image.new("RGB", (1, 1)).save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        calls = [
            tool_call("tavily_search_tool", {"query": "sunglasses"}, "one"),
            tool_call("product_catalog_tool", {}, "two"),
        ]
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            response("Looking up trends", calls),
            response("Aviators are popular."),
            response('{"prompt": "Aviator sunglasses on a beach", "caption": "Summer style"}'),
            response('{"quote": "See summer", "justification": "Fits the trend"}'),
            response("Polished trend summary"),
        ]
        client.images.generate.return_value = SimpleNamespace(data=[SimpleNamespace(b64_json=image_b64)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "campaign image.png"
            report = root / "campaign.md"
            with (
                patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}, clear=True),
                patch.object(example1.tools, "handle_tool_call", side_effect=[[{"title": "trend"}], [{"name": "Aviator"}]]) as dispatch,
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                result = example1.run_sunglasses_campaign_pipeline(client, "chat-model", "image-model", image, report)
            self.assertEqual(dispatch.call_count, 2)
            self.assertEqual(client.chat.completions.create.call_count, 5)
            self.assertEqual(client.images.generate.call_args.kwargs["model"], "image-model")
            self.assertEqual(image.read_bytes(), buffer.getvalue())
            text = report.read_text(encoding="utf-8")
            self.assertIn("![Campaign visual](campaign%20image.png)", text)
            self.assertIn("See summer", text)
            self.assertEqual(result["visual"]["caption"], "Summer style")
            self.assertNotIn("IPython.core.display", output.getvalue())
            messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
            self.assertEqual([msg["tool_call_id"] for msg in messages if msg["role"] == "tool"], ["one", "two"])

    def test_existing_output_fails_before_network_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "existing.png"
            image.write_bytes(b"existing")
            client = MagicMock()
            with self.assertRaises(FileExistsError):
                example1.run_sunglasses_campaign_pipeline(client, "model", "image", image, Path(directory) / "report.md")
            client.chat.completions.create.assert_not_called()
            self.assertEqual(image.read_bytes(), b"existing")

    def test_research_tool_turn_limit_is_enforced(self):
        client = MagicMock()
        client.chat.completions.create.return_value = response(tool_calls=[
            tool_call("product_catalog_tool", {}, "one")
        ])
        with patch.object(example1.tools, "handle_tool_call", return_value=[]), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "within 2 turns"):
                example1.market_research_agent(client, "model", max_turns=2)
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_malformed_model_response_is_reported(self):
        with self.assertRaisesRegex(ValueError, "caption"):
            example1.json_response(response('{"prompt": "only"}'), ("prompt", "caption"))

    def test_tavily_tool_dispatch_and_retry_failure(self):
        call = tool_call("tavily_search_tool", {"query": "summer"}, "one")
        with (
            patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}, clear=True),
            patch.object(example1.tools, "TavilyClient") as tavily,
            patch.object(example1.tools.time, "sleep"),
        ):
            tavily.return_value.search.return_value = {"results": [{"title": "Trend", "url": "https://example.test"}]}
            result = example1.tools.handle_tool_call(call)
            self.assertEqual(result[0]["title"], "Trend")
            tavily.return_value.search.side_effect = example1.tools.TavilyTimeoutError(1)
            with self.assertRaises(example1.tools.TavilyTimeoutError):
                example1.tools.handle_tool_call(call)
            self.assertEqual(tavily.return_value.search.call_count, 4)


if __name__ == "__main__":
    unittest.main()
