import argparse
import base64
import json
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from PIL import Image

import tools


def create_client() -> OpenAI:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api_key = os.getenv("AI_CREDIT_KEY")
    base_url = os.getenv("AI_CREDIT_URL")
    if not api_key or not api_key.strip():
        raise ValueError("Set AI_CREDIT_KEY to your AI Credit API key.")
    if not base_url or not base_url.strip():
        raise ValueError("Set AI_CREDIT_URL to your AI Credit API base URL.")
    return OpenAI(api_key=api_key.strip(), base_url=base_url.strip())


def response_text(response: Any) -> str:
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("AI Credit returned a response without text.")
    return content.strip()


def json_response(response: Any, required: tuple[str, ...]) -> dict[str, str]:
    result = json.loads(response_text(response))
    if not isinstance(result, dict) or any(
        not isinstance(result.get(key), str) or not result[key].strip() for key in required
    ):
        raise ValueError(f"AI Credit response must contain non-empty strings: {', '.join(required)}.")
    return {key: result[key] for key in required}


def market_research_agent(client: OpenAI, model: str, max_turns: int = 5) -> str:
    print("\nMarket Research Agent")
    prompt = f"""
You are a fashion market research agent preparing a trend analysis for a summer sunglasses campaign.
Use web search to explore current sunglasses trends, review the product catalog, and recommend
catalog products that match the trends. Today is {datetime.now().strftime("%Y-%m-%d")}.
Summarize the top 2-3 trends, the matching products, and why they fit the campaign.
"""
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]
    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools.get_available_tools(),
            tool_choice="auto",
        )
        message = response.choices[0].message
        if message.tool_calls:
            calls = []
            for call in message.tool_calls:
                if call.type != "function":
                    raise ValueError("AI Credit requested an unsupported tool call.")
                calls.append({
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                })
            messages.append({"role": "assistant", "content": message.content, "tool_calls": calls})
            for call in message.tool_calls:
                print(f"Tool call: {call.function.name}")
                result = tools.handle_tool_call(call)
                print(f"Tool result: {json.dumps(result, ensure_ascii=False)}")
                messages.append(tools.create_tool_response_message(call, result))
        elif message.content and message.content.strip():
            print(message.content.strip())
            return message.content.strip()
        else:
            raise ValueError("AI Credit returned neither text nor tool calls.")
    raise RuntimeError(f"Market research did not finish within {max_turns} turns.")


def graphic_designer_agent(
    client: OpenAI, trend_insights: str, model: str, image_path: Path, image_model: str
) -> dict[str, str]:
    print("\nGraphic Designer Agent")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Write an image-generation prompt and a short punchy marketing caption."},
            {"role": "user", "content": f"Trend insights:\n{trend_insights}\nReturn JSON with prompt and caption."},
        ],
        response_format={"type": "json_object"},
    )
    visual = json_response(response, ("prompt", "caption"))
    image_response = client.images.generate(
        model=image_model, prompt=visual["prompt"], size="1024x1024", quality="medium", n=1
    )
    if not image_response.data or not image_response.data[0].b64_json:
        raise ValueError("AI Credit returned no image data.")
    image_bytes = base64.b64decode(image_response.data[0].b64_json, validate=True)
    with Image.open(BytesIO(image_bytes)) as image:
        if image.format != "PNG":
            raise ValueError("AI Credit returned an image that is not PNG.")
        image.verify()
    with image_path.open("xb") as image_file:
        image_file.write(image_bytes)
    print(f"Image saved: {image_path}")
    print(f"Caption: {visual['caption']}")
    return {"image_path": str(image_path), "prompt": visual["prompt"], "caption": visual["caption"]}


def copywriter_agent(client: OpenAI, image_path: Path, trend_summary: str, model: str) -> dict[str, str]:
    print("\nCopywriter Agent")
    b64_img = base64.b64encode(image_path.read_bytes()).decode("ascii")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Create an elegant campaign quote based on an image and trend summary."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}", "detail": "auto"}},
                {"type": "text", "text": (
                    f"Trend summary:\n{trend_summary}\n"
                    'Return JSON with "quote" (at most 12 words) and "justification".'
                )},
            ]},
        ],
        response_format={"type": "json_object"},
    )
    result = json_response(response, ("quote", "justification"))
    print(f"Quote: {result['quote']}")
    return result


def packaging_agent(
    client: OpenAI, trend_summary: str, image_path: Path, quote_text: str,
    justification: str, output_path: Path, model: str
) -> str:
    print("\nPackaging Agent")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Write elegant campaign summaries for executives."},
            {"role": "user", "content": f"Rewrite this trend summary for a CEO audience:\n{trend_summary}"},
        ],
    )
    summary = response_text(response)
    relative_image = os.path.relpath(image_path.resolve(), output_path.resolve().parent)
    image_link = quote(Path(relative_image).as_posix(), safe="/")
    markdown_content = f"""# Summer Sunglasses Campaign - Executive Summary

## Refined Trend Insights
{summary}

## Campaign Visual
![Campaign visual]({image_link})

## Campaign Quote
{quote_text.strip()}

## Why This Works
{justification.strip()}

---
*Report generated on {datetime.now().strftime('%Y-%m-%d')}*
"""
    with output_path.open("x", encoding="utf-8") as report:
        report.write(markdown_content)
    print(f"Report saved: {output_path}")
    return str(output_path)


def run_sunglasses_campaign_pipeline(
    client: OpenAI, model: str, image_model: str, image_path: Path, output_path: Path
) -> dict[str, Any]:
    if image_path.resolve() == output_path.resolve():
        raise ValueError("Image and report paths must be different.")
    if image_path.suffix.lower() != ".png":
        raise ValueError("The image output path must end in .png.")
    try:
        os.path.relpath(image_path.resolve(), output_path.resolve().parent)
    except ValueError as exc:
        raise ValueError("Image and report must be on the same drive.") from exc
    for path in (image_path, output_path):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")
        if not path.parent.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {path.parent}")
    if not os.getenv("TAVILY_API_KEY", "").strip():
        raise ValueError("Set TAVILY_API_KEY to your Tavily API key.")

    trend_summary = market_research_agent(client, model)
    visual = graphic_designer_agent(client, trend_summary, model, image_path, image_model)
    copy = copywriter_agent(client, image_path, trend_summary, model)
    report_path = packaging_agent(
        client, trend_summary, image_path, copy["quote"], copy["justification"], output_path, model
    )
    return {"trend_summary": trend_summary, "visual": visual, "quote": copy, "markdown_path": report_path}


def main() -> None:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser(description="Generate a sunglasses campaign using AI Credit and Tavily.")
    parser.add_argument("--model", default="gpt-5-mini", help="AI Credit chat and vision model.")
    parser.add_argument("--image-model", default="gpt-image-1-mini", help="AI Credit image model.")
    parser.add_argument("--image", type=Path, default=Path(f"generated_image_{stamp}.png"), help="New PNG path.")
    parser.add_argument("--output", type=Path, default=Path(f"campaign_summary_{stamp}.md"), help="New report path.")
    args = parser.parse_args()
    with create_client() as client:
        run_sunglasses_campaign_pipeline(client, args.model, args.image_model, args.image, args.output)


if __name__ == "__main__":
    main()
