import argparse
import json
from datetime import datetime
from typing import Literal

import qrcode
import requests
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
)
from qrcode.image.styledpil import StyledPilImage

import utils


def get_current_time() -> str:
    """Return the current local time."""
    return datetime.now().strftime("%H:%M:%S")

def get_weather_from_ip() -> str:
    """
    Gets the current, high, and low temperature in Fahrenheit for the user's
    location and returns it to the user.
    """
    # Get location coordinates from the IP address
    location_response = requests.get("https://ipinfo.io/json", timeout=15)
    location_response.raise_for_status()
    lat, lon = location_response.json()["loc"].split(",")

    # Set parameters for the weather API call
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m",
        "daily": "temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit",
        "timezone": "auto"
    }

    # Get weather data
    weather_response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
    weather_response.raise_for_status()
    weather_data = weather_response.json()

    # Format and return the simplified string
    return (
        f"Current: {weather_data['current']['temperature_2m']}°F, "
        f"High: {weather_data['daily']['temperature_2m_max'][0]}°F, "
        f"Low: {weather_data['daily']['temperature_2m_min'][0]}°F"
    )

# Write a text file
def write_txt_file(file_path: str, content: str) -> str:
    """
    Write a string into a new text file.
    Args:
        file_path (str): Destination path.
        content (str): Text to write.
    Returns:
        str: Path to the written file.
    """
    with open(file_path, "x", encoding="utf-8") as f:
        f.write(content)
    return file_path


# Create a QR code
def generate_qr_code(data: str, filename: str, image_path: str) -> str:
    """Generate a QR code image given data and an image path.

    Args:
        data: Text or URL to encode
        filename: Name for the output PNG file (without extension)
        image_path: Path to the image to be used in the QR code
    """
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(data)

    img = qr.make_image(image_factory=StyledPilImage, embedded_image_path=image_path)
    output_file = f"{filename}.png"
    with open(output_file, "xb") as output:
        img.save(output, format="PNG")

    return f"QR code saved as {output_file} containing: {data[:50]}..."


def run_tool(name: str, arguments: object, tool_mode: Literal["single", "multiple"]) -> str:
    if not isinstance(arguments, dict):
        raise ValueError(f"Arguments for {name} must be a JSON object.")
    if name == "get_current_time":
        if arguments:
            raise ValueError("get_current_time does not accept arguments.")
        return get_current_time()
    if tool_mode == "multiple" and name == "get_weather_from_ip":
        if arguments:
            raise ValueError("get_weather_from_ip does not accept arguments.")
        return get_weather_from_ip()
    if tool_mode == "multiple" and name == "write_txt_file":
        if set(arguments) != {"file_path", "content"} or not all(
            isinstance(arguments[key], str) for key in ("file_path", "content")
        ):
            raise ValueError("write_txt_file requires string file_path and content arguments.")
        return write_txt_file(arguments["file_path"], arguments["content"])
    if tool_mode == "multiple" and name == "generate_qr_code":
        if set(arguments) != {"data", "filename", "image_path"} or not all(
            isinstance(arguments[key], str) for key in ("data", "filename", "image_path")
        ):
            raise ValueError("generate_qr_code requires string data, filename, and image_path arguments.")
        return generate_qr_code(arguments["data"], arguments["filename"], arguments["image_path"])
    raise ValueError(f"The model requested an unsupported tool: {name}")


def run_conversation(
    client: OpenAI,
    prompt: str = "What time is it?",
    tool_mode: Literal["single", "multiple"] = "single",
    model: str = "gpt-4o",
) -> str:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": prompt}
    ]
    tools: list[ChatCompletionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current local time.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        }
    ]
    if tool_mode == "multiple":
        tools.extend([
            {
                "type": "function",
                "function": {
                    "name": "get_weather_from_ip",
                    "description": "Get current, high, and low temperatures for the user's IP location.",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_txt_file",
                    "description": "Write a new text file. Existing files are not overwritten.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Destination file path."},
                            "content": {"type": "string", "description": "Text to write."},
                        },
                        "required": ["file_path", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_qr_code",
                    "description": "Create a new QR code PNG with an embedded image. Existing files are not overwritten.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {"type": "string", "description": "Text or URL to encode."},
                            "filename": {"type": "string", "description": "Output filename without .png."},
                            "image_path": {"type": "string", "description": "Path of the image to embed."},
                        },
                        "required": ["data", "filename", "image_path"],
                        "additionalProperties": False,
                    },
                },
            },
        ])
    for _ in range(5):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            if not message.content:
                raise ValueError("The model returned neither a response nor a tool call.")
            return message.content

        calls: list[ChatCompletionMessageFunctionToolCallParam] = []
        results: list[tuple[str, str]] = []
        for call in message.tool_calls:
            if call.type != "function":
                raise ValueError("The model requested an unsupported tool.")
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError as exc:
                raise ValueError("The model returned invalid tool arguments.") from exc
            result = run_tool(call.function.name, arguments, tool_mode)
            print(f"Tool called: {call.function.name}")
            print(f"Tool result: {result}")
            calls.append({
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            })
            results.append((call.id, result))

        assistant_message: ChatCompletionAssistantMessageParam = {
            "role": "assistant",
            "content": message.content,
            "tool_calls": calls,
        }
        messages.append(assistant_message)
        for call_id, result in results:
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

    raise RuntimeError("The model did not complete the conversation within five turns.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate single or multiple LLM tools.")
    parser.add_argument("--demo", choices=("single", "multiple"), default="single")
    parser.add_argument("--prompt", help="Question or task for the assistant.")
    parser.add_argument("--model", default="gpt-4o", help="AI Credit model identifier.")
    args = parser.parse_args()
    prompt = args.prompt
    if prompt is None:
        prompt = (
            "What time is it?"
            if args.demo == "single"
            else "Use the tools to tell me the current time and weather at my location."
        )
    with utils._get_openai_client() as client:
        print(run_conversation(client, prompt, args.demo, args.model))

if __name__ == "__main__":
    main()
