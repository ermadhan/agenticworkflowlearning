import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI
from PIL import Image

import tools_example1


def completion(message):
    return {
        "id": "test-completion",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
    }


def tool_message(name="get_current_time", arguments="{}"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }],
    }


class ToolConversationTests(unittest.TestCase):
    def test_tool_result_is_sent_to_the_configured_gateway(self):
        requests = []

        def handle(request):
            requests.append(request)
            message = tool_message() if len(requests) == 1 else {
                "role": "assistant", "content": "It is 12:34:56."
            }
            return httpx.Response(200, json=completion(message))

        with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
            with OpenAI(api_key="test-key", base_url="https://gateway.example/v1", http_client=http_client) as client:
                with patch("tools_example1.get_current_time", return_value="12:34:56"):
                    self.assertEqual(tools_example1.run_conversation(client), "It is 12:34:56.")

        self.assertEqual(len(requests), 2)
        self.assertEqual(str(requests[0].url), "https://gateway.example/v1/chat/completions")
        self.assertEqual(requests[0].headers["authorization"], "Bearer test-key")
        initial = json.loads(requests[0].content)
        self.assertEqual(initial["tools"][0]["function"]["name"], "get_current_time")
        second = json.loads(requests[1].content)
        self.assertEqual(second["messages"][-1], {
            "role": "tool", "tool_call_id": "call-1", "content": "12:34:56"
        })

    def test_multiple_tool_calls_are_dispatched_and_results_returned(self):
        requests = []
        with tempfile.TemporaryDirectory() as directory:
            logo = Path(directory) / "logo.png"
            note = Path(directory) / "note.txt"
            qr = Path(directory) / "qr"
            Image.new("RGB", (20, 20), "white").save(logo)
            calls = [
                ("time", "get_current_time", {}),
                ("weather", "get_weather_from_ip", {}),
                ("note", "write_txt_file", {"file_path": str(note), "content": "Reminder"}),
                ("qr", "generate_qr_code", {"data": "https://example.com", "filename": str(qr), "image_path": str(logo)}),
            ]

            def handle(request):
                requests.append(request)
                message = {
                    "role": "assistant", "content": None,
                    "tool_calls": [
                        {"id": call_id, "type": "function",
                         "function": {"name": name, "arguments": json.dumps(arguments)}}
                        for call_id, name, arguments in calls
                    ],
                } if len(requests) == 1 else {"role": "assistant", "content": "Done."}
                return httpx.Response(200, json=completion(message))

            output = io.StringIO()
            with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
                with OpenAI(api_key="test-key", base_url="https://gateway.example/v1", http_client=http_client) as client:
                    with (
                        patch("tools_example1.get_current_time", return_value="12:34:56"),
                        patch("tools_example1.get_weather_from_ip", return_value="Sunny"),
                        contextlib.redirect_stdout(output),
                    ):
                        self.assertEqual(tools_example1.run_conversation(client, "Use all tools", "multiple"), "Done.")

            self.assertEqual(note.read_text(encoding="utf-8"), "Reminder")
            self.assertTrue(qr.with_suffix(".png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertIn("Tool called: get_weather_from_ip", output.getvalue())
            initial = json.loads(requests[0].content)
            self.assertEqual({tool["function"]["name"] for tool in initial["tools"]}, {name for _, name, _ in calls})
            messages = json.loads(requests[1].content)["messages"]
            self.assertEqual([msg["tool_call_id"] for msg in messages if msg["role"] == "tool"],
                             [call_id for call_id, _, _ in calls])

    def test_tools_do_not_overwrite_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            note = Path(directory) / "note.txt"
            note.write_text("original", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                tools_example1.run_tool("write_txt_file", {"file_path": str(note), "content": "changed"}, "multiple")
            self.assertEqual(note.read_text(encoding="utf-8"), "original")

            logo = Path(directory) / "logo.png"
            Image.new("RGB", (20, 20), "white").save(logo)
            image = Path(directory) / "qr.png"
            image.write_bytes(b"original")
            with self.assertRaises(FileExistsError):
                tools_example1.run_tool("generate_qr_code", {
                    "data": "example", "filename": str(image.with_suffix("")), "image_path": str(logo)
                }, "multiple")
            self.assertEqual(image.read_bytes(), b"original")

    def test_rejects_unknown_tools_and_invalid_arguments(self):
        for message in (tool_message("other"), tool_message(arguments="bad"), tool_message(arguments='{"x": 1}')):
            with self.subTest(message=message):
                transport = httpx.MockTransport(lambda request: httpx.Response(200, json=completion(message)))
                with httpx.Client(transport=transport) as http_client:
                    with OpenAI(api_key="test-key", base_url="https://gateway.example/v1", http_client=http_client) as client:
                        with self.assertRaises(ValueError):
                            tools_example1.run_conversation(client)

    def test_stops_after_five_tool_turns(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=completion(tool_message())))
        with httpx.Client(transport=transport) as http_client:
            with OpenAI(api_key="test-key", base_url="https://gateway.example/v1", http_client=http_client) as client:
                with self.assertRaisesRegex(RuntimeError, "five turns"):
                    tools_example1.run_conversation(client)


if __name__ == "__main__":
    unittest.main()
