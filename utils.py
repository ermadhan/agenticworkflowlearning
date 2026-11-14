# === Standard Library ===
import os
import re
import base64
import mimetypes
import sys

# === Third-Party ===
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

# === Env & Clients ===
load_dotenv()


def _get_ai_credit_config() -> tuple[str, str]:
    api_key = os.getenv("AI_CREDIT_KEY")
    base_url = os.getenv("AI_CREDIT_URL")
    if not api_key or not api_key.strip():
        raise ValueError("Set AI_CREDIT_KEY to your AI Credit API key.")
    if not base_url or not base_url.strip():
        raise ValueError("Set AI_CREDIT_URL to your AI Credit API base URL.")
    return api_key.strip(), base_url.strip()


def _get_openai_client() -> OpenAI:
    api_key, base_url = _get_ai_credit_config()
    return OpenAI(api_key=api_key, base_url=base_url)


def _get_anthropic_client() -> Anthropic:
    return Anthropic()


def get_response(model: str, prompt: str) -> str:
    if "claude" in model.lower() or "anthropic" in model.lower():
        # Anthropic Claude format
        message = _get_anthropic_client().messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        return message.content[0].text

    else:
        response = _get_openai_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("AI Credit returned a response without text content.")
        return content
    
# === Data Loading ===
def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and derive date parts commonly used in charts."""
    df = pd.read_csv(csv_path)
    # Be tolerant if 'date' exists
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["quarter"] = df["date"].dt.quarter
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
    return df

# === Helpers ===
def make_schema_text(df: pd.DataFrame) -> str:
    """Return a human-readable schema from a DataFrame."""
    return "\n".join(f"- {c}: {dt}" for c, dt in df.dtypes.items())

def ensure_execute_python_tags(text: str) -> str:
    """Normalize code to be wrapped in <execute_python>...</execute_python>."""
    text = text.strip()
    # Strip ```python fences if present
    text = re.sub(r"^```(?:python)?\s*|\s*```$", "", text).strip()
    if "<execute_python>" not in text:
        text = f"<execute_python>\n{text}\n</execute_python>"
    return text

def encode_image_b64(path: str) -> tuple[str, str]:
    """Return (media_type, base64_str) for an image file path."""
    mime, _ = mimetypes.guess_type(path)
    media_type = mime or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return media_type, b64


def image_anthropic_call(model_name: str, prompt: str, media_type: str, b64: str) -> str:
    """
    Call Anthropic Claude (messages.create) with text+image and return *all* text blocks concatenated.
    Returns the model's text response.
    """
    msg = _get_anthropic_client().messages.create(
        model=model_name,
        max_tokens=2000,
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
            ],
        }],
    )

    # Anthropic returns a list of content blocks; collect all text
    parts = []
    for block in (msg.content or []):
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def image_openai_call(model_name: str, prompt: str, media_type: str, b64: str) -> str:
    data_url = f"data:{media_type};base64,{b64}"
    resp = _get_openai_client().chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )
    content = resp.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("AI Credit returned a response without text content.")
    content = content.strip()
    return content

# === Execution of LLM-generated code ===
import traceback as _traceback


def extract_code(text: str) -> str:
    """Return the code inside <execute_python>...</execute_python> (or the raw text if no tags)."""
    m = re.search(r"<execute_python>([\s\S]*?)</execute_python>", text)
    code = m.group(1) if m else text
    code = re.sub(r"^```(?:python)?\s*|\s*```$", "", code.strip()).strip()
    return code


def run_generated_code(code: str, df: pd.DataFrame, model: str = "gpt-4o-mini", max_retries: int = 1) -> str:
    """
    Execute LLM-generated plotting code with a fresh namespace containing `df`.
    If it raises, send the code + traceback back to the model, ask for a corrected
    version and retry (at most `max_retries` times). Returns the code that finally ran.
    Raises the final error if all attempts fail. This does not sandbox generated code.
    """
    code = extract_code(code)
    for attempt in range(max_retries + 1):
        try:
            exec(code, {"df": df})
            return code
        except Exception:
            tb = _traceback.format_exc()
            if attempt == max_retries:
                raise
            print(f"Generated code failed (attempt {attempt + 1}); asking the model to fix it:\n{tb}", file=sys.stderr)
            fix_prompt = f"""
The following Python/matplotlib code raised an error when executed.
Fix the code so it runs without errors and produces the same chart.

Rules:
- Keep the same intent, title, labels and output path.
- Do NOT use ax.bar_label / plt.bar_label; if value labels are needed use ax.text with f-strings.
- Only use plain format strings like f"{{v:,.0f}}" (no spaces inside format specifiers).
- Use pandas/matplotlib only. Assume df already exists. Do not call plt.show(); call plt.close() at the end.
- Return ONLY the corrected code wrapped in <execute_python></execute_python> tags, no prose.

Error:
{tb}

Code:
{code}
"""
            code = extract_code(get_response(model, fix_prompt))
