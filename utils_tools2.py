# ================================
# Imports
# ================================

# --- Standard library ---
import os
import json
from collections.abc import Callable
from datetime import datetime  # keep if used later
from urllib.parse import urljoin, urlsplit

# --- Third-party ---
import requests
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
)
from typing import Any

# --- Local / project ---
import email_tools
from utils import _get_openai_client

# ================================
# Environment & HTTP session
# ================================
load_dotenv()  # loads .env from the working directory

BASE_URL = os.getenv("M3_EMAIL_SERVER_API_URL")

session = requests.Session()
session.headers.update({"User-Agent": "LF-ADP-EmailClient/1.0"})


def _email_server_url() -> str:
    if not BASE_URL or not BASE_URL.strip():
        raise ValueError("Set M3_EMAIL_SERVER_API_URL to the email server URL in .env.")
    base = BASE_URL.strip().rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("M3_EMAIL_SERVER_API_URL must be an absolute HTTP or HTTPS URL.")
    return base


# ================================
# Helpers
# ================================
def print_html(content: Any, title: str | None = None, is_image: bool = False):
    """Print content in a readable form for a terminal."""
    if title:
        print(f"\n{title}:")
    if is_image:
        print(f"Image: {content}")
    elif isinstance(content, pd.DataFrame):
        print(content.to_string(index=False))
    elif isinstance(content, pd.Series):
        print(content.to_string())
    else:
        print(content)

def pretty_display(title: str, response: requests.Response):
    """Print an HTTP response and return its parsed content when possible."""
    status = response.status_code
    try:
        content = response.json()
        body = json.dumps(content, indent=2)
    except ValueError:
        content = response.text
        body = content

    print(f"\n{title}: HTTP {status}")
    print(body)
    return content

# ================================
# API calls
# ================================
def reset_database() -> dict:
    """Calls the /reset_database endpoint and returns the confirmation message."""
    r = session.get(f"{_email_server_url()}/reset_database")
    r.raise_for_status()
    return r.json()

def test_send_email():
    payload = {
        "recipient": "test@example.com",
        "subject": "Test Subject",
        "body": "This is a test email body.",
    }
    r = session.post(f"{_email_server_url()}/send", json=payload)
    return pretty_display("POST /send", r)

def test_list_emails():
    r = session.get(f"{_email_server_url()}/emails")
    return pretty_display("GET /emails", r)

def test_search_emails(q: str = "report"):
    r = session.get(f"{_email_server_url()}/emails/search", params={"q": q})
    return pretty_display(f"GET /emails/search?q={q}", r)

def test_filter_emails(recipient: str | None = None, date_from: str | None = None, date_to: str | None = None):
    params = {}
    if recipient:
        params["recipient"] = recipient
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    r = session.get(f"{_email_server_url()}/emails/filter", params=params)
    return pretty_display("GET /emails/filter", r)

def test_unread_emails():
    r = session.get(f"{_email_server_url()}/emails/unread")
    return pretty_display("GET /emails/unread", r)

def test_get_email(email_id: str):
    r = session.get(f"{_email_server_url()}/emails/{email_id}")
    return pretty_display(f"GET /emails/{email_id}", r)

def test_mark_read(email_id: str):
    r = session.patch(f"{_email_server_url()}/emails/{email_id}/read")
    return pretty_display(f"PATCH /emails/{email_id}/read", r)

def test_mark_unread(email_id: str):
    r = session.patch(f"{_email_server_url()}/emails/{email_id}/unread")
    return pretty_display(f"PATCH /emails/{email_id}/unread", r)

def test_delete_email(email_id: str):
    r = session.delete(f"{_email_server_url()}/emails/{email_id}")
    return pretty_display(f"DELETE /emails/{email_id}", r)


def run_email_agent(
    client: OpenAI,
    prompt: str,
    model: str,
    include_delete: bool = False,
    max_turns: int = 5,
) -> str:
    functions: dict[str, Callable[..., Any]] = {
        "search_unread_from_sender": email_tools.search_unread_from_sender,
        "list_unread_emails": email_tools.list_unread_emails,
        "search_emails": email_tools.search_emails,
        "get_email": email_tools.get_email,
        "mark_email_as_read": email_tools.mark_email_as_read,
        "send_email": email_tools.send_email,
    }
    parameters: dict[str, dict[str, str]] = {
        "search_unread_from_sender": {"sender": "string"},
        "list_unread_emails": {},
        "search_emails": {"query": "string"},
        "get_email": {"email_id": "integer"},
        "mark_email_as_read": {"email_id": "integer"},
        "send_email": {"recipient": "string", "subject": "string", "body": "string"},
        "delete_email": {"email_id": "integer"},
    }
    if include_delete:
        functions["delete_email"] = email_tools.delete_email

    tools: list[ChatCompletionToolParam] = []
    for name, function in functions.items():
        properties = {key: {"type": kind} for key, kind in parameters[name].items()}
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": (function.__doc__ or name).strip().splitlines()[0],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            },
        })

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]
    for _ in range(max_turns):
        response = client.chat.completions.create(model=model, messages=messages, tools=tools)
        message = response.choices[0].message
        if not message.tool_calls:
            if not message.content:
                raise ValueError("The model returned neither a response nor a tool call.")
            return message.content

        calls: list[ChatCompletionMessageFunctionToolCallParam] = []
        results: list[tuple[str, str]] = []
        for call in message.tool_calls:
            if call.type != "function" or call.function.name not in functions:
                raise ValueError("The model requested an unavailable email tool.")
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(f"The model returned invalid arguments for {name}.") from exc
            expected = parameters[name]
            if not isinstance(arguments, dict) or set(arguments) != set(expected):
                raise ValueError(f"The model returned invalid arguments for {name}.")
            for key, kind in expected.items():
                value = arguments[key]
                if (kind == "string" and not isinstance(value, str)) or (
                    kind == "integer" and (not isinstance(value, int) or isinstance(value, bool))
                ):
                    raise ValueError(f"The model returned invalid arguments for {name}.")
            result = functions[name](**arguments)
            print(f"Tool called: {name}")
            print(json.dumps(result, indent=2))
            calls.append({
                "id": call.id,
                "type": "function",
                "function": {"name": name, "arguments": call.function.arguments},
            })
            results.append((call.id, json.dumps(result)))

        assistant_message: ChatCompletionAssistantMessageParam = {
            "role": "assistant", "content": message.content, "tool_calls": calls
        }
        messages.append(assistant_message)
        for call_id, result in results:
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

    raise RuntimeError(f"The model did not complete the conversation within {max_turns} turns.")


def call_llm_email_agent(prompt: str,
                         api_url: str | None = None,
                         timeout: int = 30) -> dict:
    """
    Calls the M3 LLM server with a natural-language instruction.

    Args:
        prompt: Instruction for the agent (e.g., "Check unread emails...").
        api_url: Base URL of the LLM server. If None, uses env var M3_LLM_SERVER_URL.
        timeout: HTTP timeout in seconds.

    Returns:
        dict with keys: ok (bool), status (int), response (str|None), raw (dict|str)
    """
    # Resolve API base URL
    base = api_url or os.getenv("M3_LLM_SERVER_URL")
    if not base:
        raise RuntimeError("M3_LLM_SERVER_URL is not set. Put it in your .env (e.g., http://127.0.0.1:5001).")

    # Build final endpoint; accept both with/without trailing /prompt
    endpoint = base if base.rstrip("/").endswith("/prompt") else urljoin(base.rstrip("/") + "/", "prompt")

    try:
        r = requests.post(endpoint, json={"prompt": prompt}, timeout=timeout)
    except requests.RequestException as e:
        return {"ok": False, "status": None, "response": None, "raw": str(e)}

    try:
        data = r.json()
    except ValueError:
        data = r.text

    ok = (r.status_code == 200)
    return {"ok": ok, "status": r.status_code, "response": (data.get("response") if isinstance(data, dict) else None), "raw": data}
