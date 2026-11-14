import importlib
import sys
import types
import unittest
from unittest.mock import patch


class ResearchClientTests(unittest.TestCase):
    def test_ai_credit_client_uses_aisuite_tool_calling(self):
        research_tools = types.ModuleType("research_tools")
        for name in ("arxiv_search_tool", "tavily_search_tool", "wikipedia_search_tool"):
            setattr(research_tools, name, lambda query: query)

        with patch.dict(sys.modules, {"research_tools": research_tools}):
            sys.modules.pop("componenteval1", None)
            try:
                example = importlib.import_module("componenteval1")
                with (
                    patch.dict(
                        "os.environ",
                        {"AI_CREDIT_KEY": " test-key ", "AI_CREDIT_URL": " https://gateway.example/v1 "},
                        clear=True,
                    ),
                    patch.object(example, "Client") as client,
                ):
                    completion = client.return_value.chat.completions.create
                    completion.return_value.choices[0].message.content = "Result"

                    self.assertEqual(example.find_references("Find papers"), "Result")
                    client.assert_called_once_with(provider_configs={
                        "openai": {"api_key": "test-key", "base_url": "https://gateway.example/v1"}
                    })
                    kwargs = completion.call_args.kwargs
                    self.assertEqual(kwargs["model"], "openai:gpt-4o")
                    self.assertEqual(kwargs["max_turns"], 5)
                    self.assertEqual(kwargs["tools"], [
                        research_tools.arxiv_search_tool,
                        research_tools.tavily_search_tool,
                        research_tools.wikipedia_search_tool,
                    ])
            finally:
                sys.modules.pop("componenteval1", None)


if __name__ == "__main__":
    unittest.main()
