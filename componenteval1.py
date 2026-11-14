# =========================
# Imports
# =========================

# --- Standard library 
from datetime import datetime
import json
import re

# --- Third-party ---
from aisuite import Client

# --- Local / project ---
import research_tools
import utils

def find_references(task: str, model: str = "openai:gpt-4o", return_messages: bool = False):
    """Perform a research task using external tools (arxiv, tavily, wikipedia)."""

    prompt = f"""
    You are a research function with access to:
    - arxiv_tool: academic papers
    - tavily_tool: general web search (return JSON when asked)
    - wikipedia_tool: encyclopedic summaries

    Task:
    {task}

    Today is {datetime.now().strftime('%Y-%m-%d')}.
    """.strip()

    messages = [{"role": "user", "content": prompt}]
    tools = [
        research_tools.arxiv_search_tool,
        research_tools.tavily_search_tool,
        research_tools.wikipedia_search_tool,
    ]

    api_key, base_url = utils._get_ai_credit_config()
    client = Client(provider_configs={"openai": {"api_key": api_key, "base_url": base_url}})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_turns=5,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("AI Credit returned a response without text content.")
    return (content, messages) if return_messages else content


# list of preferred domains for Tavily results
TOP_DOMAINS = {
    # General reference / institutions / publishers
    "wikipedia.org", "nature.com", "science.org", "sciencemag.org", "cell.com",
    "mit.edu", "stanford.edu", "harvard.edu", "nasa.gov", "noaa.gov", "europa.eu",

    # CS/AI venues & indexes
    "arxiv.org", "acm.org", "ieee.org", "neurips.cc", "icml.cc", "openreview.net",

    # Other reputable outlets
    "elifesciences.org", "pnas.org", "jmlr.org", "springer.com", "sciencedirect.com",

    # Extra domains (case-specific additions)
    "pbs.org", "nova.edu", "nvcc.edu", "cccco.edu",

    # Well known programming sites
    "codecademy.com", "datacamp.com"
}

def evaluate_tavily_results(TOP_DOMAINS, raw: str, min_ratio=0.4):
    """
    Evaluate whether plain-text research results mostly come from preferred domains.

    Args:
        TOP_DOMAINS (set[str]): Set of preferred domains (e.g., 'arxiv.org', 'nature.com').
        raw (str): Plain text or Markdown containing URLs.
        min_ratio (float): Minimum preferred ratio required to pass (e.g., 0.4 = 40%).

    Returns:
        tuple[bool, str]: (flag, markdown_report)
            flag -> True if PASS, False if FAIL
            markdown_report -> Markdown-formatted summary of the evaluation
    """

    # Extract URLs from the text
    url_pattern = re.compile(r'https?://[^\s\]\)>\}]+', flags=re.IGNORECASE)
    urls = url_pattern.findall(raw)

    if not urls:
        return False, """### Evaluation — Tavily Preferred Domains
No URLs detected in the provided text. 
Please include links in your research results.
"""

    # Count preferred vs total
    total = len(urls)
    preferred_count = 0
    details = []

    for url in urls:
        domain = url.split("/")[2]
        preferred = any(td in domain for td in TOP_DOMAINS)
        if preferred:
            preferred_count += 1
        details.append(f"- {url} → {'✅ PREFERRED' if preferred else '❌ NOT PREFERRED'}")

    ratio = preferred_count / total if total > 0 else 0.0
    flag = ratio >= min_ratio

    # Markdown report
    report = f"""
### Evaluation — Tavily Preferred Domains
- Total results: {total}
- Preferred results: {preferred_count}
- Ratio: {ratio:.2%}
- Threshold: {min_ratio:.0%}
- Status: {"✅ PASS" if flag else "❌ FAIL"}

**Details:**
{chr(10).join(details)}
"""
    return flag, report

def main():
    research_task = "Find 2 recent papers about recent developments in black hole science"
    research_result = find_references(research_task)
    print(f"Research Function Output:\n{research_result}")
    print(f"\nSample Trusted Domains:\n{json.dumps(list(TOP_DOMAINS)[:4], indent=2)}")
    print(f"\nResearch Results:\n{research_result}")
    flag, report = evaluate_tavily_results(TOP_DOMAINS, research_result)
    print(f"\nEvaluation Summary:\n{report}")

    topic = "recent developments in black hole science"
    min_ratio = 0.4
    preferred_domains = {
        "wikipedia.org", "nature.com", "science.org", "arxiv.org",
        "nasa.gov", "mit.edu", "stanford.edu", "harvard.edu"
    }

    print(f"\nSample Preferred Domains:\n{json.dumps(sorted(preferred_domains), indent=2)}")
    research_task = f"Find 2–3 key papers and reliable overviews about {topic}."
    research_output = find_references(research_task)
    print(f"\nResearch Results on {topic}:\n{research_output}")
    flag, eval_md = evaluate_tavily_results(preferred_domains, research_output, min_ratio=min_ratio)
    print(f"\nEvaluation Summary:\n{eval_md}")


if __name__ == "__main__":
    main()