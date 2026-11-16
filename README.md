# Coffee sales chart workflow

Configure `AI_CREDIT_KEY` and `AI_CREDIT_URL` in environment variables or a local
`.env` file (see `.env.example`). Use your provider's API base URL and supported
model names. Install the project dependencies from `requirements.txt` in a
virtual environment.

Run from PowerShell:

```powershell
.\venv\Scripts\python.exe .\reflectiondesignpatternexample.py
```

Use `--help` to see options for the dataset, instruction, models, and output
basename. The script writes `chart_v1.png` and `chart_v2.png` in the current
directory by default; existing files with those names may be overwritten.

The script executes model-generated Python without a sandbox. Run it only in a
trusted environment; generated code has the permissions of your user account.

# SQL reflection workflow

```powershell
.\venv\Scripts\python.exe .\reflectiondesignpatternexample_sql.py
```

The SQL example uses the same AI Credit settings. It uses `products.db` next
to the script, creating sample data only if that file does not exist. Use
`--help` to choose another database, question, or supported model. Generated
SQL runs read-only; the schema and first query's results are sent to the model
for reflection.

# Tool-calling example

Run `.\venv\Scripts\python.exe .\tools_example1.py` using the same AI Credit
settings for the single-tool time demo. For the multiple-tool demo, run
`.\venv\Scripts\python.exe .\tools_example1.py --demo multiple`. Use `--prompt`
to request a text file or a QR code with an existing image, and `--model` to
choose a model supporting OpenAI-compatible function calling.

The weather tool sends your IP address to ipinfo.io, then sends its reported
coordinates to Open-Meteo and the resulting weather to the model. The multiple
demo lets the model create text and PNG files at requested paths; existing files
are not overwritten.

# Inventory planning example

Run `.\venv\Scripts\python.exe .\M5_UGL_1\example1.py` with the AI Credit
settings above. Use `--model` for a supported model and `--prompt` to change the
final customer request. The demonstration sends prompts and sample inventory
information to the model; inventory and transactions are updated only in memory.

# Sunglasses campaign example

Set `TAVILY_API_KEY` in `.env` along with the AI Credit settings. Run
`.\venv\Scripts\python.exe .\M5_UGL_2\example1.py`; use `--help` to select
models and output paths. The provider must support chat, vision, and image
generation. The script sends a search query to Tavily and sends search results,
sample product information, and the generated image to the AI Credit provider.
It writes a generated PNG image (e.g. `M5_UGL_2/generated_image_*.png`) and a
Markdown report (`M5_UGL_2/campaign_summary_*.md`) without overwriting existing files.

# Research component evaluation

`componenteval1.py` evaluates individual agent components through AISuite with the
AI Credit settings. It uses `research_tools.py` in the root directory providing
`arxiv_search_tool`, `tavily_search_tool`, and `wikipedia_search_tool`.

# Email example

Set `M3_EMAIL_SERVER_API_URL` to the email server's base URL; the bundled
simulator can be served at `http://127.0.0.1:5000` with:

```powershell
.\venv\Scripts\python.exe -m uvicorn email_server.email_service:app --host 127.0.0.1 --port 5000
```

Startup preserves existing records and seeds sample emails only if the database
is empty. The `/reset_database` endpoint still deletes and reloads all records.
Once the server is running, run `.\venv\Scripts\python.exe .\tools_example2.py`.
The example resets the simulated database, sends and marks emails, and runs
examples that delete emails.

# C1M2 Assignment: Reflection Design Pattern

Implements an essay drafting, reflection, and revision agent workflow using AISuite (`C1M2_Assignment/assignment.py`):

- **`generate_draft(topic, model)`**: Produces an initial draft based on the topic prompt.
- **`reflect_on_draft(draft, model)`**: Analyzes the draft and provides constructive feedback and critique.
- **`revise_draft(original_draft, reflection, model)`**: Integrates the critique to produce a polished, revised version.
- **`utils.show_output()`**: Formats output for rich HTML in Jupyter Notebooks or formatted text in terminal sessions.

Run unit tests and the workflow:

```powershell
.\venv\Scripts\python.exe .\C1M2_Assignment\assignment.py
```

# C1M3 Assignment: Tool Use & Structured Reflection

Demonstrates tool-calling integration and multi-stage reflection using OpenAI-compatible function calling (`C1M3_Assignment/assignment.py`):

- **Tools Integration**: Connects arXiv search (`arxiv_search_tool`) and web search (`tavily_search_tool`).
- **`generate_research_report_with_tools(prompt, model)`**: Autonomous multi-turn loop that allows the LLM to inspect, decide, and execute tool calls over multiple turns until a sourced academic report is complete.
- **`reflection_and_rewrite(report, model)`**: Structured reviewer that returns JSON with explicit headings (`Strengths`, `Limitations`, `Suggestions`, `Opportunities`) and a rewritten report.
- **`convert_report_to_html(report, model)`**: Formats raw text/markdown research reports into clean, styled HTML documents.

Run unit tests and the pipeline:

```powershell
.\venv\Scripts\python.exe .\C1M3_Assignment\assignment.py
```

# C1M5 Assignment: Multi-Agent Research System

Implements an autonomous multi-agent planning and execution architecture using AISuite (`C1M5_Assignment/assignment.py`):

- **`planner_agent(topic, model)`**: Decomposes a research goal into an atomic list of executable task strings ending with Markdown report creation.
- **`research_agent(task, model, return_messages)`**: Research specialist utilizing callable tools (`arxiv_search_tool`, `tavily_search_tool`, `wikipedia_search_tool`) with automatic multi-turn tool calling (`max_turns=6`).
- **`writer_agent(task, model)`**: Writing specialist that drafts academic content from accumulated research context.
- **`editor_agent(task, model)`**: Editorial specialist that critiques, improves, and shapes the final Markdown deliverable.
- **`executor_agent(topic, model, limit_steps)`**: Orchestration engine that dynamically inspects each plan step, routes it to the designated agent (`research_agent`, `writer_agent`, or `editor_agent`) via structured JSON decisions, passes accumulated step history, and generates the final research report.

Run unit tests and the multi-agent pipeline:

```powershell
.\venv\Scripts\python.exe .\C1M5_Assignment\assignment.py
```
