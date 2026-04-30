# 📊 Financial News Sentiment & Summarization Dashboard

A Streamlit dashboard that fetches the 5 most recent news
headlines for any stock ticker, runs each through **GPT-4o-mini** with
equity-research-analyst prompt engineering, and surfaces sentiment scores,
key risks, and professional summaries in a clean terminal-style UI.

---

## Features

| Feature | Detail |
|---|---|
| **Real-time news** | Pulls headlines via `yfinance` |
| **AI sentiment** | GPT-4o-mini scores each headline −1 → +1 |
| **Analyst voice** | System prompt written as a bulge-bracket research desk |
| **Risk extraction** | Up to 3 forward-looking risk factors per headline |
| **Session export** | One-click CSV download of all analysed tickers |
| **Rate-limit safe** | 500 ms sleep between API calls |

---

## Quick Start

### 1. Clone / download

```bash
git clone <your-repo-url>
cd finance-sentiment-tool
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key (secure, never hard-coded)

Create a file named **`.env`** in the project root:

```
OPENAI_API_KEY=sk-proj-...your-key-here...
```

### 5. Run the dashboard

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

---

## How It Works

```
User enters ticker
       │
       ▼
yfinance.Ticker(ticker).news   ← up to 5 headlines
       │
       ▼  (for each headline, 0.5 s sleep between calls)
OpenAI GPT-4o-mini
  system: "Senior Equity Research Analyst at bulge-bracket bank…"
  user:   headline + JSON schema instruction
       │
       ▼
{ sentiment_score, key_risks, summary }
       │
       ▼
Streamlit metrics + headline cards + pandas DataFrame
       │
       ▼
Optional CSV export
```

---

## Prompt Engineering Notes

The system prompt instructs the model to write in the register of a
Goldman Sachs / Morgan Stanley published research note:

- **Tone:** Dispassionate, precise, forward-looking
- **Sentiment score:** Calibrated to *near-term price impact*, not general opinion
- **Key risks:** Actionable, institutional-grade (not consumer advice)
- **Temperature:** 0.2 — keeps outputs factual and low-variance

---


---

## Tech Stack

- **Python 3.11+**
- **Streamlit** — web UI framework
- **yfinance** — Yahoo Finance data wrapper
- **OpenAI Python SDK** — GPT-4o-mini inference
- **pandas** — tabular data management & CSV export
- **python-dotenv** — secure environment variable loading

---

## License

MIT — free to use and modify.
