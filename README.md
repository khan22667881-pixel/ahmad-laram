# ahmad-laram
# ATS Resume Checker

Upload a resume (PDF, DOCX or TXT) and get an estimated ATS score, a category
breakdown, prioritized improvements, missing keywords and rewrite examples.
Optionally paste a job description to get a job-match score.

Built with [Streamlit](https://streamlit.io) and Google Gemini Flash.

> The score is an AI-based estimate, not an official result from any real ATS.

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at https://aistudio.google.com/apikey, then either set it as an
environment variable:

```bash
export GEMINI_API_KEY="your-key"      # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```

or create `.streamlit/secrets.toml` (never commit this file):

```toml
GEMINI_API_KEY = "your-key"
# GEMINI_MODEL = "gemini-2.5-flash"   # optional
```

If no key is configured, the app shows a key box in the sidebar.

## Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | secrets or env var | none (required) |
| `GEMINI_MODEL` | secrets, env var or sidebar | `gemini-2.5-flash` |

If the model name stops working, check the current Flash model name in the
[Gemini docs](https://ai.google.dev/gemini-api/docs/models) and change `GEMINI_MODEL`.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public, or private with access granted).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick the repo and branch, and set the main file to `app.py`.
4. Open **Advanced settings > Secrets** and add: `GEMINI_API_KEY = "your-key"`
5. Click **Deploy**.

## Privacy

Resume text is sent to the Gemini API for analysis. The app does not store files.
