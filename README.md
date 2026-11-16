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
It writes a PNG and Markdown report without overwriting existing files.

# Research example

`componenteval1.py` uses the same AI Credit settings through AISuite. It also
requires a `research_tools` module providing `arxiv_search_tool`,
`tavily_search_tool`, and `wikipedia_search_tool`; that module is not included.

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
