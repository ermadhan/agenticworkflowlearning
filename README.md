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
