import contextlib
import io
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import example1
import inventory_utils


def response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class StandaloneExampleTests(unittest.TestCase):
    def test_client_uses_ai_credit_settings(self):
        with (
            patch.object(example1, "load_dotenv"),
            patch.dict("os.environ", {"AI_CREDIT_KEY": " key ", "AI_CREDIT_URL": " https://example.test/v1 "}, clear=True),
            patch.object(example1, "OpenAI") as client,
        ):
            example1.create_client()
            client.assert_called_once_with(api_key="key", base_url="https://example.test/v1")

    def test_missing_configuration_is_reported(self):
        with patch.object(example1, "load_dotenv"), patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "AI_CREDIT_KEY"):
                example1.create_client()

    def test_standalone_workflow_runs_with_mocked_model(self):
        plan = {
            "steps": [
                {"step_number": 1, "tools": [
                    {"use": "get_inventory_data", "args": {"product_name": "Aviator"}, "result_key": "prod"}
                ], "validations": [
                    {"name": "found", "use_tool": "assert_true", "args": {"value_from": "context.prod.item"}}
                ]},
                {"step_number": 2, "tools": [
                    {"use": "update_inventory", "args": {"item_id_from": "context.prod.item.item_id", "delta": 2}, "result_key": "update"},
                    {"use": "append_transaction", "args": {"customer_name": "WALK_IN", "summary": "Return 2 Aviators", "amount": -100}, "result_key": "transaction"}
                ], "validations": []},
            ]
        }
        reflection = {"critique": "Looks good", "revised_plan": plan}
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            response(json.dumps(plan)),
            response(json.dumps(reflection)),
            response("Insufficient stock."),
            response(json.dumps(plan)),
            response(json.dumps(reflection)),
        ]
        manager = MagicMock()
        manager.__enter__.return_value = client
        output = io.StringIO()
        with (
            patch.object(example1, "create_client", return_value=manager),
            patch.object(sys, "argv", ["example1.py", "--model", "test-model"]),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(example1.main(), 0)
        self.assertEqual(client.chat.completions.create.call_count, 5)
        self.assertTrue(all(call.kwargs["model"] == "test-model" for call in client.chat.completions.create.call_args_list))
        self.assertIn("Overall execution status: SUCCESS", output.getvalue())
        self.assertIn("Return 2 Aviators", output.getvalue())
        self.assertNotIn("IPython.core.display", output.getvalue())

    def test_tool_error_aborts_before_later_transaction(self):
        inventory = inventory_utils.create_inventory_dataframe()
        transactions = inventory_utils.create_transaction_dataframe()
        plan = {"steps": [
            {"step_number": 1, "tools": [
                {"use": "update_inventory", "args": {"item_id": "missing", "delta": 1}, "result_key": "update"}
            ]},
            {"step_number": 2, "tools": [
                {"use": "append_transaction", "args": {"customer_name": "A", "summary": "Sale", "amount": 1}, "result_key": "txn"}
            ]},
        ]}
        report = example1.execute_plan_tools_only(plan, inventory, transactions)
        self.assertFalse(report["ok"])
        self.assertEqual(report["abort_reason"], "tool_failed")
        self.assertIn("item_not_found", report["steps"][0]["tool_error"])
        self.assertEqual(len(report["steps"]), 1)
        self.assertEqual(len(report["updated_frames"]["transaction_df"]), 1)

    def test_return_updates_only_the_result_frames(self):
        inventory = inventory_utils.create_inventory_dataframe()
        transactions = inventory_utils.create_transaction_dataframe()
        original_quantity = int(inventory.loc[0, "quantity_in_stock"])
        plan = {"steps": [{
            "step_number": 1,
            "tools": [
                {"use": "update_inventory", "args": {"item_id": "SG001", "delta": 2}, "result_key": "update"},
                {"use": "append_transaction", "args": {"customer_name": "A", "summary": "Return", "amount": -100}, "result_key": "txn"},
            ],
        }]}
        report = example1.execute_plan_tools_only(plan, inventory, transactions)
        self.assertTrue(report["ok"])
        self.assertEqual(report["updated_frames"]["inventory_df"].loc[0, "quantity_in_stock"], original_quantity + 2)
        self.assertEqual(report["updated_frames"]["transaction_df"].iloc[-1]["transaction_amount"], -100)
        self.assertEqual(inventory.loc[0, "quantity_in_stock"], original_quantity)
        self.assertEqual(len(transactions), 1)

    def test_malformed_reflection_is_not_executed(self):
        client = MagicMock()
        client.chat.completions.create.return_value = response('{"critique": "missing plan"}')
        with self.assertRaisesRegex(ValueError, "without a revised plan"):
            example1.reflect_on_plan(client, "Return glasses", {"steps": []})


if __name__ == "__main__":
    unittest.main()
