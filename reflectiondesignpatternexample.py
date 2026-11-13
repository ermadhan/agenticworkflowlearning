import argparse
import json
import os
from pathlib import Path
import re

import utils

def generate_chart_code(instruction: str, model: str, out_path_v1: str) -> str:
    """Generate Python code to make a plot with matplotlib using tag-based wrapping."""

    prompt = f"""
    You are a data visualization expert.

    Return your answer *strictly* in this format:

    <execute_python>
    # valid python code here
    </execute_python>

    Do not add explanations, only the tags and the code.

    The code should create a visualization from a DataFrame 'df' with these columns:
    - date   (datetime64 — already parsed; use df['date'].dt.year, df['date'].dt.month, etc.)
    - time   (string, HH:MM — do NOT concatenate or combine with the date column)
    - cash_type (string: 'card' or 'cash')
    - card (string)
    - price (number)
    - coffee_name (string)
    - quarter (int, 1–4 — already computed, use directly)
    - month  (int, 1–12 — already computed, use directly)
    - year   (int, e.g. 2024 — already computed, use directly)

    User instruction: {instruction}

    Requirements for the code:
    1. Assume the DataFrame is already loaded as 'df'.
    2. Use matplotlib for plotting.
    3. Add clear title, axis labels, and legend if needed.
    4. Save the figure as '{out_path_v1}' with dpi=300.
    5. Do not call plt.show().
    6. Close all plots with plt.close().
    7. Add all necessary import python statements
    8. CRITICAL: 'date' is datetime64 — never use string concatenation on it.
       Filter by year/quarter using the 'year' and 'quarter' integer columns.
    9. Do NOT use ax.bar_label / plt.bar_label. If you want value labels, use ax.text with f-strings.
    10. Only use plain format strings such as f"{{v:,.0f}}" (no spaces inside the format specifier).
    11. Stick to basic, widely supported matplotlib functions (bar, plot, legend, title, xlabel, ylabel, tight_layout, savefig).
    
    Return ONLY the code wrapped in <execute_python> tags.
    """

    response = utils.get_response(model, prompt)
    return response

def reflect_on_image_and_regenerate(
    chart_path: str,
    instruction: str,
    model_name: str,
    out_path_v2: str,
    code_v1: str,  
) -> tuple[str, str]:
    """
    Critique the chart IMAGE and the original code against the instruction, 
    then return refined matplotlib code.
    Returns (feedback, refined_code_with_tags).
    Supports OpenAI and Anthropic (Claude).
    """
    media_type, b64 = utils.encode_image_b64(chart_path)
    

    prompt = f"""
    You are a data visualization expert.
    Your task: critique the attached chart and the original code against the given instruction,
    then return improved matplotlib code.

    Original code (for context):
    {code_v1}

    OUTPUT FORMAT (STRICT):
    1) First line: a valid JSON object with ONLY the "feedback" field.
    Example: {{"feedback": "The legend is unclear and the axis labels overlap."}}

    2) After a newline, output ONLY the refined Python code wrapped in:
    <execute_python>
    ...
    </execute_python>

    3) Import all necessary libraries in the code. Don't assume any imports from the original code.

    HARD CONSTRAINTS:
    - Do NOT include Markdown, backticks, or any extra prose outside the two parts above.
    - Use pandas/matplotlib only (no seaborn).
    - Assume df already exists; do not read from files.
    - Save to '{out_path_v2}' with dpi=300.
    - Always call plt.close() at the end (no plt.show()).
    - Include all necessary import statements.
    - Do NOT use ax.bar_label / plt.bar_label. If you want value labels, use ax.text with f-strings.
    - Only use plain format strings such as f"{{v:,.0f}}" (no spaces inside the format specifier).
    - Stick to basic, widely supported matplotlib functions.
    
    IMPORTANT: The 'date' column is already a pandas datetime64 type.
    - Do NOT concatenate 'date' with 'time' using string operations.
    - To filter by year/quarter, use: df[df['year'] == 2024] or df['date'].dt.year == 2024
    - The 'quarter' and 'year' columns already exist as integers; use them directly.

    Schema (columns available in df):
    - date   (datetime64 — already parsed; use df['date'].dt.year, etc.)
    - time   (string, HH:MM — do NOT concatenate with date)
    - cash_type (string: 'card' or 'cash')
    - card   (string)
    - price  (float)
    - coffee_name (string)
    - quarter (int, 1–4)
    - month  (int, 1–12)
    - year   (int)
    
    CRITICAL TYPE RULE: 'date' is already datetime64.
    - NEVER do: df['date'] + ' ' + df['time']  ← this will crash
    - ALWAYS filter by year/quarter using the integer columns: df[df['year'] == 2024]

    Instruction:
    {instruction}
    """


    # In case the name is "Claude" or "Anthropic", use the safe helper
    lower = model_name.lower()
    if "claude" in lower or "anthropic" in lower:
        # ✅ Use the safe helper that joins all text blocks and adds a system prompt
        content = utils.image_anthropic_call(model_name, prompt, media_type, b64)
    else:
        content = utils.image_openai_call(model_name, prompt, media_type, b64)

    # --- Parse ONLY the first JSON line (feedback) ---
    lines = content.strip().splitlines()
    json_line = lines[0].strip() if lines else ""

    try:
        obj = json.loads(json_line)
    except json.JSONDecodeError as exc:
        raise ValueError("The chart reflection did not start with valid JSON.") from exc
    if not isinstance(obj, dict) or not isinstance(obj.get("feedback"), str):
        raise ValueError("The chart reflection must contain a feedback string.")

    # --- Extract refined code from <execute_python>...</execute_python> ---
    m_code = re.search(r"<execute_python>([\s\S]*?)</execute_python>", content)
    if not m_code or not m_code.group(1).strip():
        raise ValueError("The chart reflection did not contain executable code.")
    refined_code_body = m_code.group(1).strip()
    refined_code = utils.ensure_execute_python_tags(refined_code_body)

    feedback = str(obj.get("feedback", "")).strip()
    return feedback, refined_code

def run_workflow(
    dataset_path: str,
    user_instructions: str,
    generation_model: str,
    reflection_model: str,   
    image_basename: str = "chart",
):
    """
    End-to-end pipeline:
      1) load dataset
      2) generate V1 code
      3) execute V1 → produce chart_v1.png
      4) reflect on V1 (image + original code) → feedback + refined code
      5) execute V2 → produce chart_v2.png

    Returns a dict with all artifacts (codes, feedback, image paths).
    """
    # 0) Load dataset; utils handles parsing and feature derivations (e.g., year/quarter)
    df = utils.load_and_prepare_data(dataset_path)
    print("Random sample of dataset:")
    print(df.sample(n=min(5, len(df))).to_string(index=False))

    # Paths to store charts
    out_v1 = f"{image_basename}_v1.png"
    out_v2 = f"{image_basename}_v2.png"

    # 1) Generate code (V1)
    print("\nStep 1: Generating chart code (V1)...")
    code_v1 = generate_chart_code(
        instruction=user_instructions,
        model=generation_model,
        out_path_v1=out_v1,
    )
    print("Generated code (V1):\n", code_v1, sep="")

    print("\nStep 2: Executing chart code (V1)...")
    executed_code_v1 = execute_chart(code_v1, df, generation_model, out_v1)
    print(f"Generated chart (V1): {out_v1}")

    # 3) Reflect on V1 (image + original code) to get feedback and refined code (V2)
    print("\nStep 3: Reviewing the chart and generating improvements...")
    feedback, code_v2 = reflect_on_image_and_regenerate(
        chart_path=out_v1,
        instruction=user_instructions,
        model_name=reflection_model,
        out_path_v2=out_v2,
        code_v1=executed_code_v1,
    )
    print(f"Feedback on V1:\n{feedback}")
    print(f"Generated code (V2):\n{code_v2}")

    print("\nStep 4: Executing refined chart code (V2)...")
    execute_chart(code_v2, df, reflection_model, out_v2)
    print(f"Generated chart (V2): {out_v2}")

    return {
        "code_v1": code_v1,
        "chart_v1": out_v1,
        "feedback": feedback,
        "code_v2": code_v2,
        "chart_v2": out_v2,
    }


def execute_chart(code: str, df, model: str, output_path: str) -> str:
    match = re.search(r"<execute_python>([\s\S]*?)</execute_python>", code)
    if not match or not match.group(1).strip():
        raise ValueError("The model response did not contain executable chart code.")
    path = Path(output_path)
    before = path.stat() if path.exists() else None
    executed_code = utils.run_generated_code(match.group(1).strip(), df, model=model)
    if not path.is_file():
        raise FileNotFoundError(f"Generated code did not create the chart: {path}")
    after = path.stat()
    if before and (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size):
        raise RuntimeError(f"Generated code did not update the existing chart: {path}")
    return executed_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and refine a coffee sales chart.")
    parser.add_argument("--dataset", default=str(Path(__file__).resolve().with_name("coffee_sales.csv")))
    parser.add_argument(
        "--instruction",
        default="Create a plot comparing Q1 coffee sales in 2024 and 2025 using the data in coffee_sales.csv.",
    )
    parser.add_argument("--generation-model", default="gpt-4o-mini")
    parser.add_argument("--reflection-model", default="gpt-5-mini")
    parser.add_argument("--image-basename", default="chart")
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    run_workflow(args.dataset, args.instruction, args.generation_model, args.reflection_model, args.image_basename)


if __name__ == "__main__":
    main()
