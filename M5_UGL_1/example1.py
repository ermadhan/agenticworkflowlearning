# =========================
# Imports & utilities
# =========================

# --- Standard library ---
from __future__ import annotations
from typing import Any
import argparse
import json
import os
import re
from pathlib import Path

# --- Third-party ---
import pandas as pd
import duckdb
from openai import OpenAI
from dotenv import load_dotenv

# --- Local ---
import inventory_utils
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


def print_output(content: Any, title: str) -> None:
    print(f"\n{title}:")
    if isinstance(content, pd.DataFrame):
        print(content.to_string(index=False))
    elif isinstance(content, pd.Series):
        print(content.to_string())
    else:
        print(content)

# =========================
# DuckDB helpers
# =========================
def create_duckdb_with_views(inventory_df: pd.DataFrame, transaction_df: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.register("inventory_df", inventory_df)
    con.register("transaction_df", transaction_df)
    return con

def execute_plan_tools_only(
    plan: dict[str, Any],
    inventory_df: pd.DataFrame,
    transaction_df: pd.DataFrame,
    return_updated_frames: bool = True,
    stop_on_failed_validation: bool = True  # <-- new flag
) -> dict[str, Any]:
    """
    Executes a plan based ONLY on tools (no SQL visible in the plan).
    - Runs tools step by step and stores results in the context.
    - Performs validations using tools.
    - Write tools return updated DataFrames and the executor applies them automatically.
    - If stop_on_failed_validation=True, execution halts at the first failed validation.
    """
    con = create_duckdb_with_views(inventory_df, transaction_df)
    ctx: dict[str, Any] = {
        "__con__": con,
        "__frames__": {"inventory_df": inventory_df.copy(), "transaction_df": transaction_df.copy()}
    }

    report: dict[str, Any] = {"ok": True, "steps": []}
    try:
        for step in plan.get("steps", []):
            step_number = step.get("step_number")
            description = step.get("description", "")

            # tools
            tool_error = None
            try:
                ran = tools.run_tools_for_step(step, ctx)
            except Exception as e:
                ran = {}
                tool_error = str(e)
                report["ok"] = False

            # validations
            validations = [tools.run_tool_validation(v, ctx) for v in step.get("validations", [])]
            step_ok = (tool_error is None) and all(v.get("ok", False) for v in validations)
            if not step_ok:
                report["ok"] = False

            # record the step
            report["steps"].append({
                "step_number": step_number,
                "description": description,
                "tools_run": list(ran.keys()),
                "tool_error": tool_error,
                "validations": validations,
            })

            # stop execution if a validation failed
            if stop_on_failed_validation and not step_ok:
                report["aborted"] = True
                report["abort_step"] = step_number
                report["abort_reason"] = "tool_failed" if tool_error else "validation_failed"
                break

    finally:
        con.close()

    if return_updated_frames:
        report["updated_frames"] = {
            "inventory_df": ctx["__frames__"]["inventory_df"],
            "transaction_df": ctx["__frames__"]["transaction_df"],
        }

    return report

# Minimal one-step plan: lookup the product "Aviator"
simple_plan = {
    "reasoning": "User wants to check availability of Aviator sunglasses.",
    "steps": [
        {
            "step_number": 1,
            "description": "Lookup Aviator sunglasses in inventory",
            "tools": [
                {"use": "get_inventory_data", "args": {"product_name": "Aviator"}, "result_key": "prod"}
            ],
            "validations": [
                {"name": "product_found", "use_tool": "assert_true", "args": {"value_from": "context.prod.item"}}
            ]
        }
    ]
}

# =========================
# Planning spec (TOOLS-ONLY) and planning workflow
# =========================

# Shared planning spec: TOOLS ONLY (no raw SQL in the plan)
PLANNING_SPEC_TOOLS_ONLY = """
You are a planning system for a sunglasses store. Produce a FULL, AUTONOMOUS plan using TOOLS ONLY.
We will run this plan against two pandas DataFrames registered in DuckDB as views:
- inventory_df(name, item_id, description, quantity_in_stock, price)
- transaction_df(transaction_id, customer_name, transaction_summary, transaction_amount, balance_after_transaction)

Customer intents include:
- Purchase: "I want to buy 3 Aviators"
- Return: "I'd like to return two Sport sunglasses"
- Inquiry: "Do you have Mystique glasses?"
- Browse: "Show me what's available"

IMPORTANT: ALLOWED TOOLS ONLY (do NOT invent new tools)
Tool catalog (names, exact args, outputs):
1) get_inventory_data
   - args: { product_name?: string, item_id?: string }
   - returns: { rows: DataFrame, match_count: int, item: dict|null }
   - notes: Use this for product lookup (case-insensitive by name) or by item_id.
2) get_transaction_data
   - args: { mode?: "last_balance" }
   - returns: { mode: string, last_txn_id: string|null, last_balance: number }
3) compute_total
   - args: { qty: number, price: number }
   - returns: { amount: number }
4) compute_refund
   - args: { qty: number, price: number }   # Refund is negative by design
   - returns: { amount: number }
5) update_inventory
   - args: { item_id: string, delta?: number, quantity_new?: number }
   - returns: { inventory_df: DataFrame, updated: { item_id: string, quantity_in_stock: number } }
   - notes: For purchase use delta = -qty. For return use delta = +qty.
6) append_transaction
   - args: { customer_name: string, summary: string, amount: number }
   - returns: { transaction_df: DataFrame, transaction: { ... } }
7) assert_true
   - args: { value: any }                    # passes if truthy (non-null/non-zero/non-empty)
   - returns: { ok: boolean }
8) assert_nonnegative_stock
   - args: { inventory_df: DataFrame, item_id: string }
   - returns: { ok: boolean, qty: number }

STRICT RULES:
1) Return VALID JSON ONLY with keys: reasoning, steps.
2) Each step MUST contain:
   - "step_number": integer
   - "description": short human text
   - "tools": an array of tool calls in order. Each tool call is:
       {"use": "<tool_name>", "args": {...}, "result_key": "<context_key>"}
     * You MAY reference previous results using dotted paths starting with "context.", e.g., "context.prod.item.price".
     * Use *_from to resolve from context, e.g., {"price_from": "context.prod.item.price"}.
     * Use ONLY the tools listed above. Do NOT use names like assert_one, assert_gt, assert_contains, format_return_summary, lookup_product, propose_transaction, etc.
     * Strings like the transaction summary MUST be composed inline by you (e.g., "Return 2 Sport sunglasses").
   - "validations": array of tool validations:
       {"name": "...", "use_tool": "<tool_name>", "args": {...}}
     * Allowed validation tools: assert_true, assert_nonnegative_stock ONLY.
     * Examples:
         - product_found: assert_true with {"value_from": "context.prod.item"} (non-null)
         - nonnegative_stock_after_update: assert_nonnegative_stock with {"inventory_df_from": "context.__frames__.inventory_df", "item_id_from": "context.prod.item.item_id"}
3) Do NOT include raw SQL in the plan. Tools run any needed SQL internally.
4) For purchases/returns, include tool calls to:
   - Lookup product via get_inventory_data (case-insensitive by name)
   - (Purchase only) compute_total; (Return only) compute_refund
   - Create a clear summary STRING inline (e.g., "Purchase 3 Aviator sunglasses" / "Return 2 Sport sunglasses")
   - Update inventory via update_inventory (delta = -qty for purchases, +qty for returns)
   - Append the transaction via append_transaction (amount from compute_total/compute_refund)
5) Use canonical arg names exactly as in the Tool catalog:
   - quantity -> use qty
   - unit_price -> use price
   - Do NOT add extra args like sign; compute_refund already returns negative amounts.

OUTPUT JSON SHAPE:
{
  "reasoning": "...",
  "steps": [
    {
      "step_number": 1,
      "description": "...",
      "tools": [ {"use": "...", "args": {...}, "result_key": "..."} ],
      "validations": [ {"name": "...", "use_tool": "...", "args": {...}} ]
    }
  ]
}

EXAMPLE (Return 2 Sport sunglasses for a walk-in):
{
  "reasoning": "User requests a return of 2 Sport units. We'll lookup product, compute a negative refund, update stock (+2), and append a refund transaction.",
  "steps": [
    {
      "step_number": 1,
      "description": "Lookup product 'Sport' and capture item details",
      "tools": [
        {"use": "get_inventory_data", "args": {"product_name": "Sport"}, "result_key": "prod"}
      ],
      "validations": [
        {"name": "product_found", "use_tool": "assert_true", "args": {"value_from": "context.prod.item"}}
      ]
    },
    {
      "step_number": 2,
      "description": "Compute refund amount for qty=2",
      "tools": [
        {"use": "compute_refund", "args": {"qty": 2, "price_from": "context.prod.item.price"}, "result_key": "refund"}
      ],
      "validations": []
    },
    {
      "step_number": 3,
      "description": "Update inventory by adding returned quantity",
      "tools": [
        {"use": "update_inventory", "args": {"item_id_from": "context.prod.item.item_id", "delta": 2}, "result_key": "inv_after"}
      ],
      "validations": [
        {"name": "stock_nonnegative", "use_tool": "assert_nonnegative_stock",
         "args": {"inventory_df_from": "context.__frames__.inventory_df", "item_id_from": "context.prod.item.item_id"}}
      ]
    },
    {
      "step_number": 4,
      "description": "Append the refund transaction for a walk-in customer",
      "tools": [
        {"use": "append_transaction",
         "args": {
           "customer_name": "WALK_IN_CUSTOMER",
           "summary": "Return 2 Sport sunglasses",
           "amount_from": "context.refund.amount"
         },
         "result_key": "txn"}
      ],
      "validations": [
        {"name": "transaction_created", "use_tool": "assert_true",
         "args": {"value_from": "context.txn.transaction.transaction_id"}}
      ]
    }
  ]
}
"""

def generate_plan(client: OpenAI, user_query: str, model: str = "gpt-5-mini") -> dict[str, Any]:
    context = f"{PLANNING_SPEC_TOOLS_ONLY}\n\nCustomer query: {user_query}\nProduce the plan now."
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return JSON ONLY following the TOOLS-ONLY planning spec."},
            {"role": "user", "content": context},
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    if not content:
        raise ValueError("AI Credit returned a planning response without text.")
    plan = json.loads(content)
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
        raise ValueError("AI Credit returned a plan without a steps array.")
    return plan

# =========================
# Reflection step (enforces the same TOOLS-ONLY spec)
# =========================
_ALLOWED_ESC = r'["\\/bfnrtu]'
def _repair_invalid_json_escapes(s: str) -> str:
    s = s.replace("\\'", "'")
    return re.sub(rf'\\(?!{_ALLOWED_ESC})', r'', s)

def _parse_json_or_repair(s: str) -> dict[str, Any]:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return json.loads(_repair_invalid_json_escapes(s))

def reflect_on_plan(client: OpenAI, user_query: str, draft_plan: dict[str, Any], model: str = "gpt-5-mini") -> dict[str, Any]:
    sys = (
        "You are a senior plan reviewer. Return STRICT JSON with keys "
        "'critique' (string) and 'revised_plan' (object). The revised_plan MUST follow the TOOLS-ONLY spec."
    )
    user = (
        "TOOLS-ONLY PLANNING SPEC (enforce exactly):\n"
        f"{PLANNING_SPEC_TOOLS_ONLY}\n\n"
        "Customer query:\n"
        f"{user_query}\n\n"
        "Draft plan (JSON):\n"
        f"{json.dumps(draft_plan, ensure_ascii=False)}\n\n"
        "Task: Critique the draft against the spec and return a corrected 'revised_plan' if needed. "
        "Ensure valid JSON and that no raw SQL appears in the plan (only tool calls)."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    if not content:
        raise ValueError("AI Credit returned a reflection response without text.")
    data = _parse_json_or_repair(content)
    if not isinstance(data, dict) or not isinstance(data.get("revised_plan"), dict):
        raise ValueError("AI Credit returned a reflection without a revised plan.")
    if not isinstance(data["revised_plan"].get("steps"), list):
        raise ValueError("AI Credit returned a revised plan without a steps array.")
    return data

# A purposely imperfect draft plan (arg mismatch + missing validation)
draft_plan = {
    "reasoning": "User wants to buy 2 Aviators.",
    "steps": [
        {
            "step_number": 1,
            "description": "Lookup product 'Aviator'",
            "tools": [
                {"use": "get_inventory_data", "args": {"product_name": "Aviator"}, "result_key": "prod"}
            ],
            "validations": []  # (missing product_found validation)
        },
        {
            "step_number": 2,
            "description": "Compute total for purchase",
            "tools": [
                # <-- wrong arg name: should be qty
                {"use": "compute_total", "args": {"quantity": 2, "price_from": "context.prod.item.price"}, "result_key": "total"}
            ],
            "validations": []
        }
    ]
}

def explain_execution_error(client: OpenAI, user_query: str, execution_report: dict[str, Any], model: str = "gpt-5-mini") -> str:
    sys = (
        "You are a senior plan reviewer. Given a json with the report, "
        "explain in simple terms what went wrong and how to fix it."
    )
    user = (
        "Customer query:\n"
        f"{user_query}\n\n"
        "Execution report (JSON):\n"
        f"{execution_report}\n\n"
        "Task: Explain in simple terms what went wrong and how to fix it."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": user}],
        response_format={"type": "text"},
    )
    content = resp.choices[0].message.content
    if not content:
        raise ValueError("AI Credit returned an explanation without text.")
    return content

# Example: simulate a failed execution report
failed_report = {
    "ok": False,
    "steps": [
        {
            "step_number": 1,
            "description": "Lookup product 'Aviator'",
            "tools_run": ["get_inventory_data"],
            "tool_error": None,
            "validations": [{"name": "product_found", "ok": True}],
        },
        {
            "step_number": 2,
            "description": "Update inventory by subtracting 999 units",
            "tools_run": ["update_inventory"],
            "tool_error": None,
            "validations": [{"name": "stock_nonnegative", "ok": False, "qty": -995}],
        },
    ],
    "aborted": True,
    "abort_step": 2,
    "abort_reason": "validation_failed",
}

def main() -> int:
    parser = argparse.ArgumentParser(description="Run the sunglasses store planning demonstration.")
    parser.add_argument("--model", default="gpt-5-mini", help="AI Credit model identifier.")
    parser.add_argument(
        "--prompt",
        default="I'd like to return two Aviator sunglasses for a walk-in customer",
        help="Customer request for the final workflow.",
    )
    args = parser.parse_args()

    inventory_df = inventory_utils.create_inventory_dataframe()
    transaction_df = inventory_utils.create_transaction_dataframe()
    print_output(inventory_df.head(), "Inventory DataFrame")
    print_output(transaction_df.head(), "Transaction DataFrame")

    con = create_duckdb_with_views(inventory_df, transaction_df)
    try:
        print_output(con.sql("SELECT * FROM inventory_df").df().head(), "Inventory DataFrame via SQL")
        prod = tools.TOOL_REGISTRY["get_inventory_data"](con=con, product_name="Aviator")
        print_output(prod, "Product Lookup")
        total = tools.TOOL_REGISTRY["compute_total"](qty=3, price=prod["item"]["price"])
        print_output(total, "Purchase Total")
    finally:
        con.close()

    report = execute_plan_tools_only(simple_plan, inventory_df, transaction_df)
    print_output(report, "Execution Report: Single-step Plan")

    with create_client() as client:
        query = "I'd like to return two Aviator sunglasses"
        plan = generate_plan(client, query, model=args.model)
        print_output(json.dumps(plan, indent=2), "Draft Plan from Planning Step")

        query = "I'd like to buy 2 Aviator sunglasses for a walk-in customer"
        reflection = reflect_on_plan(client, query, draft_plan, model=args.model)
        print_output(reflection.get("critique"), "Reflection Critique")
        print_output(json.dumps(reflection["revised_plan"], indent=2), "Revised Plan (TOOLS-ONLY compliant)")

        explanation = explain_execution_error(client, "Buy 999 Aviator sunglasses", failed_report, model=args.model)
        print_output(explanation, "Error Explanation")

        print_output(args.prompt, "User Prompt")
        print_output(inventory_df, "Initial Inventory (sample)")
        print_output(transaction_df, "Initial Transactions (tail)")
        plan = generate_plan(client, args.prompt, model=args.model)
        print_output(json.dumps(plan, indent=2), "Draft Plan")
        reflection = reflect_on_plan(client, args.prompt, plan, model=args.model)
        print_output(reflection.get("critique"), "Reflection Critique")
        final_plan = reflection["revised_plan"]
        print_output(json.dumps(final_plan, indent=2), "Final Plan (After Reflection)")

        report = execute_plan_tools_only(final_plan, inventory_df, transaction_df)
        print_output(f"Overall execution status: {'SUCCESS' if report['ok'] else 'FAILED'}", "Execution Status")
        for step in report["steps"]:
            print_output(step, f"Execution Report Step {step['step_number']}: {step['description']}")

        if report["ok"]:
            print_output(report["updated_frames"]["inventory_df"], "Updated Inventory")
            print_output(report["updated_frames"]["transaction_df"], "Updated Transactions")
            return 0

        print_output("The plan failed; updated data was not applied.", "Execution Status")
        error = explain_execution_error(client, args.prompt, report, model=args.model)
        print_output(error, "Error Explanation")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
