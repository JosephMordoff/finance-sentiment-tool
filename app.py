"""
Financial News Sentiment & Summarization Dashboard
====================================================
A Streamlit application that fetches recent news headlines for a given
stock ticker, analyzes sentiment using OpenAI GPT-4o-mini, and presents
results in a professional equity-research-style dashboard.

Usage:
    streamlit run app.py

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key (set in .env file)
"""

import time
import json
import os

import pandas as pd
import streamlit as st
import yfinance as yf
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Equity Sentiment Desk",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — refined dark terminal aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0d0f14;
        color: #c9d1d9;
    }
    .main { background-color: #0d0f14; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

    .stButton > button {
        background: linear-gradient(135deg, #1a8cff 0%, #0057b8 100%);
        color: white;
        border: none;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        padding: 0.55rem 1.5rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.75rem;
    }
    .metric-card .label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.15em;
        color: #8b949e;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        line-height: 1;
    }
    .score-positive { color: #3fb950; }
    .score-neutral  { color: #d29922; }
    .score-negative { color: #f85149; }

    .headline-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-left: 3px solid #1a8cff;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .headline-text {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #8b949e;
        margin-bottom: 0.5rem;
    }
    .summary-text {
        font-size: 0.92rem;
        color: #c9d1d9;
        line-height: 1.6;
    }
    .risk-list {
        font-size: 0.85rem;
        color: #f0883e;
        line-height: 1.8;
        margin-top: 0.5rem;
    }

    .stTextInput input {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1rem !important;
        border-radius: 4px !important;
    }
    .stTextInput input:focus {
        border-color: #1a8cff !important;
        box-shadow: 0 0 0 2px rgba(26,140,255,0.15) !important;
    }

    div[data-testid="stDownloadButton"] button {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
    }

    .ticker-badge {
        display: inline-block;
        background: #1a8cff22;
        border: 1px solid #1a8cff55;
        color: #1a8cff;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
    }
    hr { border-color: #21262d; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# OpenAI client (lazy-init to surface config errors clearly)
# ---------------------------------------------------------------------------
def get_openai_client() -> OpenAI:
    """
    Instantiate and return an OpenAI client using the API key from the
    environment.  Raises a RuntimeError with a user-friendly message if the
    key is absent.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. "
            "Create a .env file with OPENAI_API_KEY=sk-... and restart."
        )
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------
def analyze_sentiment(headline: str, client: OpenAI) -> dict:
    """
    Send a news headline to GPT-4o-mini and return a structured sentiment
    analysis in the voice of a professional equity research analyst.

    Parameters
    ----------
    headline : str
        A single news headline string to analyse.
    client : OpenAI
        An authenticated OpenAI client instance.

    Returns
    -------
    dict
        A dictionary with keys:
        - ``sentiment_score`` (float): value from -1.0 (bearish) to +1.0 (bullish)
        - ``key_risks``       (list[str]): bullet-point risk factors
        - ``summary``         (str): ≤40-word professional summary

    Raises
    ------
    ValueError
        If the API response cannot be parsed as valid JSON.
    """
    system_prompt = (
        "You are a Senior Equity Research Analyst at a bulge-bracket investment bank. "
        "You write in the precise, measured language found in Goldman Sachs or Morgan Stanley "
        "research notes. When given a news headline, you assess its market implications "
        "dispassionately and return ONLY a valid JSON object — no preamble, no markdown fences."
    )

    user_prompt = f"""Analyse the following equity news headline and return a JSON object with exactly these keys:

{{
  "sentiment_score": <float between -1.0 (strongly bearish) and 1.0 (strongly bullish)>,
  "key_risks": [<up to 3 concise risk factors as plain strings, no bullet symbols>],
  "summary": "<professional analyst summary in 40 words or fewer>"
}}

Headline: \"{headline}\"

Rules:
- sentiment_score must reflect the likely near-term price impact on the underlying equity.
- key_risks must be forward-looking and actionable.
- summary must read like a sentence from a published research note.
- Return ONLY the JSON object. No extra text."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=300,
    )

    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse model response as JSON: {raw}") from exc

    # Normalise: ensure key_risks is always a list
    if isinstance(result.get("key_risks"), str):
        result["key_risks"] = [result["key_risks"]]

    return result


# ---------------------------------------------------------------------------
# yfinance helpers
# ---------------------------------------------------------------------------
def fetch_news(ticker: str) -> list[dict]:
    """
    Fetch up to 5 recent news items for the given ticker using yfinance.

    Parameters
    ----------
    ticker : str
        A valid stock ticker symbol (e.g. 'AAPL', 'MSFT').

    Returns
    -------
    list[dict]
        A list of news item dicts from yfinance, capped at 5.

    Raises
    ------
    ValueError
        If the ticker is invalid or no news is available.
    """
    tk = yf.Ticker(ticker.upper().strip())
    news = tk.news
    if not news:
        raise ValueError(
            f"No news found for '{ticker}'. "
            "Check that the ticker is valid and try again."
        )
    return news[:5]


def get_company_name(ticker: str) -> str:
    """
    Return the company long name for a ticker, falling back to the ticker
    symbol itself if the info is unavailable.

    Parameters
    ----------
    ticker : str
        A valid stock ticker symbol.

    Returns
    -------
    str
        Human-readable company name or the uppercased ticker.
    """
    try:
        info = yf.Ticker(ticker.upper().strip()).info
        return info.get("longName") or ticker.upper()
    except Exception:
        return ticker.upper()


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def score_css_class(score: float) -> str:
    """Return a CSS class name based on the sentiment score magnitude."""
    if score >= 0.15:
        return "score-positive"
    if score <= -0.15:
        return "score-negative"
    return "score-neutral"


def score_label(score: float) -> str:
    """Return a human-readable label for a sentiment score."""
    if score >= 0.4:
        return "Bullish"
    if score >= 0.15:
        return "Mildly Bullish"
    if score <= -0.4:
        return "Bearish"
    if score <= -0.15:
        return "Mildly Bearish"
    return "Neutral"


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def render_headline_card(idx: int, item: dict, analysis: dict) -> None:
    """
    Render a single headline + analysis card using custom HTML.

    Parameters
    ----------
    idx : int
        1-based card index for display purposes.
    item : dict
        Raw yfinance news item containing at least a 'title' key.
    analysis : dict
        Output of ``analyze_sentiment`` for this headline.
    """
    score = analysis.get("sentiment_score", 0)
    css   = score_css_class(score)
    label = score_label(score)
    risks = analysis.get("key_risks", [])
    risks_html = "".join(f"<div>▸ {r}</div>" for r in risks)

    content = item.get("content", {})
    title = content.get("title") if isinstance(content, dict) else None
    if not title:
        title = item.get("title", "No title available")

    st.markdown(
        f"""
        <div class="headline-card">
            <div class="headline-text">#{idx} &nbsp;|&nbsp; {title}</div>
            <div class="summary-text">{analysis.get('summary', '—')}</div>
            <div class="risk-list">{risks_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for the Streamlit dashboard."""

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='font-size:1.8rem; margin-bottom:0;'>📊 Equity Sentiment Desk</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#8b949e; font-size:0.85rem; margin-top:0.25rem;'>"
        "AI-powered news sentiment analysis · powered by GPT-4o-mini</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Input row ────────────────────────────────────────────────────────────
    col_input, col_btn, col_spacer = st.columns([3, 1, 4])
    with col_input:
        ticker = st.text_input(
            "Ticker Symbol",
            placeholder="e.g. AAPL, MSFT, NVDA",
            label_visibility="collapsed",
        )
    with col_btn:
        run = st.button("ANALYSE")

    # ── Session state init ───────────────────────────────────────────────────
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame()

    # ── Analysis pipeline ────────────────────────────────────────────────────
    if run:
        if not ticker.strip():
            st.warning("Please enter a ticker symbol.")
            st.stop()

        try:
            client = get_openai_client()
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

        with st.spinner(f"Fetching news for **{ticker.upper()}** …"):
            try:
                news_items = fetch_news(ticker)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

        company = get_company_name(ticker)
        st.markdown(
            f"<div class='ticker-badge'>{ticker.upper()} · {company}</div>",
            unsafe_allow_html=True,
        )

        rows        = []
        analyses    = []
        progress    = st.progress(0, text="Analysing headlines…")

        for i, item in enumerate(news_items):
            content = item.get("content", {})
            title = content.get("title") if isinstance(content, dict) else None
            if not title:
                title = item.get("title", "No title available")
            try:
                result = analyze_sentiment(title, client)
            except (ValueError, Exception) as exc:
                st.warning(f"Headline {i+1} skipped — {exc}")
                result = {"sentiment_score": 0, "key_risks": [], "summary": "Analysis unavailable."}

            analyses.append(result)
            rows.append({
                "ticker":          ticker.upper(),
                "company":         company,
                "headline":        title,
                "sentiment_score": result.get("sentiment_score", 0),
                "sentiment_label": score_label(result.get("sentiment_score", 0)),
                "summary":         result.get("summary", ""),
                "key_risks":       " | ".join(result.get("key_risks", [])),
            })
            progress.progress((i + 1) / len(news_items), text=f"Analysed {i+1}/{len(news_items)}")
            time.sleep(0.5)

        progress.empty()

        df = pd.DataFrame(rows)
        st.session_state.results_df = pd.concat(
            [st.session_state.results_df, df], ignore_index=True
        )

        # ── Summary metrics ──────────────────────────────────────────────────
        avg_score  = df["sentiment_score"].mean()
        max_score  = df["sentiment_score"].max()
        min_score  = df["sentiment_score"].min()

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)

        def metric_card(col, label, value, css_cls):
            with col:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="label">{label}</div>
                        <div class="value {css_cls}">{value:+.2f}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        metric_card(m1, "Avg Sentiment Score",  avg_score,  score_css_class(avg_score))
        metric_card(m2, "Highest Score",         max_score,  score_css_class(max_score))
        metric_card(m3, "Lowest Score",          min_score,  score_css_class(min_score))

        overall_label = score_label(avg_score)
        overall_css   = score_css_class(avg_score)
        st.markdown(
            f"<p style='font-family:IBM Plex Mono,monospace; font-size:0.85rem; color:#8b949e;'>"
            f"Overall signal: <span class='{overall_css}'><b>{overall_label}</b></span>"
            f" &nbsp;(mean score {avg_score:+.2f})</p>",
            unsafe_allow_html=True,
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='font-size:1rem; letter-spacing:0.08em;'>HEADLINE BREAKDOWN</h3>",
            unsafe_allow_html=True,
        )

        for idx, (item, analysis) in enumerate(zip(news_items, analyses), start=1):
            render_headline_card(idx, item, analysis)

    # ── Export panel (always visible once results exist) ─────────────────────
    if not st.session_state.results_df.empty:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='font-size:1rem; letter-spacing:0.08em;'>EXPORT SESSION DATA</h3>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            st.session_state.results_df,
            use_container_width=True,
            hide_index=True,
        )
        csv_bytes = st.session_state.results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇  Download CSV",
            data=csv_bytes,
            file_name="sentiment_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
