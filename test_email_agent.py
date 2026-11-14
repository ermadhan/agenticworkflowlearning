import contextlib
import io
import json
import unittest
from unittest.mock import patch

import httpx
import requests
from openai import OpenAI

import email_tools
import utils_tools2


def completion(message):
    return {
        "id": "test",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4.1",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
    }


def calls_message(*calls):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function",
             "function": {"name": name, "arguments": json.dumps(arguments)}}
            for call_id, name, arguments in calls
        ],
    }


class EmailAgentTests(unittest.TestCase):
    def test_runs_multiple_tool_turns_and_returns_tool_results(self):
        requests_seen = []

        def handle(request):
            requests_seen.append(json.loads(request.content))
            if len(requests_seen) == 1:
                message = calls_message(("c1", "search_unread_from_sender", {"sender": "boss@example.com"}))
            elif len(requests_seen) == 2:
                message = calls_message(
                    ("c2", "mark_email_as_read", {"email_id": 7}),
                    ("c3", "send_email", {"recipient": "boss@example.com", "subject": "Re", "body": "Thanks"}),
                )
            else:
                message = {"role": "assistant", "content": "Done."}
            return httpx.Response(200, json=completion(message))

        with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
            with OpenAI(api_key="test-key", base_url="https://gateway.example/v1", http_client=http_client) as client:
                with (
                    patch("email_tools.search_unread_from_sender", return_value=[{"id": 7}]) as search,
                    patch("email_tools.mark_email_as_read", return_value={"id": 7, "read": True}) as mark,
                    patch("email_tools.send_email", return_value={"id": 8}) as send,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(utils_tools2.run_email_agent(client, "Check emails", "gpt-4.1"), "Done.")

        search.assert_called_once_with(sender="boss@example.com")
        mark.assert_called_once_with(email_id=7)
        send.assert_called_once_with(recipient="boss@example.com", subject="Re", body="Thanks")
        self.assertNotIn("max_turns", requests_seen[0])
        self.assertNotIn("delete_email", {item["function"]["name"] for item in requests_seen[0]["tools"]})
        messages = requests_seen[-1]["messages"]
        tool_results = {item["tool_call_id"]: json.loads(item["content"]) for item in messages if item["role"] == "tool"}
        self.assertEqual(tool_results, {"c1": [{"id": 7}], "c2": {"id": 7, "read": True}, "c3": {"id": 8}})

    def test_delete_is_only_available_when_enabled(self):
        request_bodies = []

        def handle(request):
            request_bodies.append(json.loads(request.content))
            message = calls_message(("delete", "delete_email", {"email_id": 4})) if len(request_bodies) == 1 else {
                "role": "assistant", "content": "Deleted."
            }
            return httpx.Response(200, json=completion(message))

        with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
            with OpenAI(api_key="test-key", base_url="https://gateway.example/v1", http_client=http_client) as client:
                with patch("email_tools.delete_email", return_value={"message": "deleted"}) as delete:
                    with self.assertRaisesRegex(ValueError, "unavailable"):
                        utils_tools2.run_email_agent(client, "Delete", "gpt-4.1")
                    delete.assert_not_called()
                    request_bodies.clear()
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(utils_tools2.run_email_agent(client, "Delete", "gpt-4.1", include_delete=True),
                                         "Deleted.")
                    delete.assert_called_once_with(email_id=4)
        self.assertIn("delete_email", {item["function"]["name"] for item in request_bodies[0]["tools"]})

    def test_invalid_tool_argument_does_not_execute(self):
        message = calls_message(("c1", "mark_email_as_read", {"email_id": True}))
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=completion(message)))
        with httpx.Client(transport=transport) as http_client:
            with OpenAI(api_key="test-key", base_url="https://gateway.example/v1", http_client=http_client) as client:
                with patch("email_tools.mark_email_as_read") as mark:
                    with self.assertRaisesRegex(ValueError, "invalid arguments"):
                        utils_tools2.run_email_agent(client, "Mark", "gpt-4.1")
                    mark.assert_not_called()

    def test_tool_loop_stops_at_five_turns(self):
        requests_seen = []

        def handle(request):
            requests_seen.append(request)
            return httpx.Response(200, json=completion(calls_message(("c1", "list_unread_emails", {}))))

        with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
            with OpenAI(api_key="test-key", base_url="https://gateway.example/v1", http_client=http_client) as client:
                with patch("email_tools.list_unread_emails", return_value=[]), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "5 turns"):
                        utils_tools2.run_email_agent(client, "List", "gpt-4.1")
        self.assertEqual(len(requests_seen), 5)

    def test_email_http_errors_are_not_reported_as_success(self):
        response = requests.Response()
        response.status_code = 404
        response._content = b'{"detail":"not found"}'
        with patch("email_tools.requests.get", return_value=response):
            with self.assertRaises(requests.HTTPError):
                email_tools.get_email(7)


if __name__ == "__main__":
    unittest.main()
