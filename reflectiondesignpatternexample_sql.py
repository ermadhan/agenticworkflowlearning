import argparse
import json
from pathlib import Path

import utils
import utils_sql
import pandas as pd

def generate_sql(question: str, schema: str, model: str) -> str:
    prompt = f"""
    You are a SQL assistant. Given the schema and the user's question, write a SQL query for SQLite.

    Schema:
    {schema}

    User question:
    {question}

    Respond with the SQL only.
    """
    response = utils.get_response(model, prompt).strip()
    if not response:
        raise ValueError("The model returned an empty SQL query.")
    return response

def refine_sql(
    question: str,
    sql_query: str,
    schema: str,
    model: str,
) -> tuple[str, str]:
    """
    Reflect on whether a query's *shown output* answers the question,
    and propose an improved SQL if needed.
    Returns (feedback, refined_sql).
    """
    prompt = f"""
You are a SQL reviewer and refiner.

User asked:
{question}

Original SQL:
{sql_query}

Table Schema:
{schema}

Step 1: Briefly evaluate if the SQL OUTPUT fully answers the user's question.
Step 2: If improvement is needed, provide a refined SQL query for SQLite.
If the original SQL is already correct, return it unchanged.

Return STRICT JSON with two fields:
{{
  "feedback": "<1-3 sentences explaining the gap or confirming correctness>",
  "refined_sql": "<final SQL to run>"
}}
"""
    return parse_refinement(utils.get_response(model, prompt))

def refine_sql_external_feedback(
    question: str,
    sql_query: str,
    df_feedback: pd.DataFrame,
    schema: str,
    model: str,
) -> tuple[str, str]:
    """
    Evaluate whether the SQL result answers the user's question and,
    if necessary, propose a refined version of the query.
    Returns (feedback, refined_sql).
    """
    prompt = f"""
    You are a SQL reviewer and refiner.

    User asked:
    {question}

    Original SQL:
    {sql_query}

    SQL Output:
    {df_feedback.to_markdown(index=False)}

    Table Schema:
    {schema}

    Step 1: Briefly evaluate if the SQL output answers the user's question.
    Step 2: If the SQL could be improved, provide a refined SQL query.
    If the original SQL is already correct, return it unchanged.

    Return a strict JSON object with two fields:
    - "feedback": brief evaluation and suggestions
    - "refined_sql": the final SQL to run
    """

    return parse_refinement(utils.get_response(model, prompt))


def parse_refinement(content: str) -> tuple[str, str]:
    content = content.strip()
    if content.startswith("```json") and content.endswith("```"):
        content = content[len("```json"):-len("```")].strip()
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("The model returned invalid JSON for SQL refinement.") from exc
    if not isinstance(obj, dict):
        raise ValueError("SQL refinement must be a JSON object.")
    feedback = obj.get("feedback")
    refined_sql = obj.get("refined_sql")
    if not isinstance(feedback, str) or not feedback.strip():
        raise ValueError("SQL refinement must include feedback.")
    if not isinstance(refined_sql, str) or not refined_sql.strip():
        raise ValueError("SQL refinement must include a query.")
    return feedback.strip(), refined_sql.strip()

def run_sql_workflow(
    db_path: str,
    question: str,
    model_generation: str = "gpt-4.1",
    model_evaluation: str = "gpt-4.1",
):
    """
    End-to-end workflow to generate, execute, evaluate, and refine SQL queries.

    Steps:
      1) Extract database schema
      2) Generate SQL (V1)
      3) Execute V1 → show output
      4) Reflect on V1 with execution feedback → propose refined SQL (V2)
      5) Execute V2 → show final answer
    """

    # 1) Schema
    schema = utils_sql.get_schema(db_path)
    print(f"Question: {question}\n\nStep 1 - Database schema:\n{schema}")

    # 2) Generate SQL (V1)
    sql_v1 = generate_sql(question, schema, model_generation)
    print(f"\nStep 2 - Generated SQL (V1):\n{sql_v1}")

    # 3) Execute V1
    df_v1 = utils_sql.execute_sql(sql_v1, db_path)
    print(f"\nStep 3 - SQL output (V1):\n{df_v1.to_string(index=False)}")

    # 4) Reflect on V1 with execution feedback → refine to V2
    feedback, sql_v2 = refine_sql_external_feedback(
        question=question,
        sql_query=sql_v1,
        df_feedback=df_v1,          # external feedback: real output of V1
        schema=schema,
        model=model_evaluation,
    )
    print(f"\nStep 4 - Feedback:\n{feedback}")
    print(f"\nStep 4 - Refined SQL (V2):\n{sql_v2}")

    # 5) Execute V2
    df_v2 = utils_sql.execute_sql(sql_v2, db_path)
    print(f"\nStep 5 - SQL output (V2):\n{df_v2.to_string(index=False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and refine a SQLite query.")
    parser.add_argument("--database", default=str(Path(__file__).resolve().with_name("products.db")))
    parser.add_argument("--question", default="Which color of product has the highest total sales?")
    parser.add_argument("--generation-model", default="gpt-4.1")
    parser.add_argument("--evaluation-model", default="gpt-4.1")
    args = parser.parse_args()

    if not Path(args.database).exists():
        utils_sql.create_transactions_db(args.database)
    run_sql_workflow(args.database, args.question, args.generation_model, args.evaluation_model)


if __name__ == "__main__":
    main()