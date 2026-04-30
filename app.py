"""
Financial News Market Sentiment 
====================================================
A Streamlit application that fetches recent news headlines for a given
stock ticker, analyzes sentiment using OpenAI GPT-4o-mini, and presents
results in a professional equity-research-style dashboard.

Usage:
    streamlit run app.py

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key (set in .env file)
"""

# from curses import raw
import time
import json
import os
from unittest import result

from altair import value
import pandas as pd
import streamlit as st
import yfinance as yf
from openai import OpenAI
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing import List

class SentimentResponse(BaseModel):
    sentiment_score: float = Field(ge=-1, le=1)
    key_risks: List[str] = Field(max_items=3)
    summary: str = Field(max_length=200)

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
        background: linear-gradient(135deg, #6001D2 0%, #4a00a8 100%);
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
        border-left: 3px solid #6001D2;
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
        background: #6001D2 !important;
        border: 1px solid #4a00a8 !important;
        color: #ffffff !important;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
    }

    .ticker-badge {
        display: inline-block;
        background: #6001D222;
        border: 1px solid #6001D255;
        color: #6001D2;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
    }
    hr { border-color: #7B2FBE; }

    .stProgress > div > div > div > div {
        background-color: #6001D2 !important;
    }

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
- Base analysis ONLY on the headline. Do not infer unstated facts.
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
        parsed = SentimentResponse.model_validate_json(raw)
        result = parsed.model_dump()
    except Exception as exc:
        raise ValueError(f"Invalid structured response: {raw}") from exc
    
    return result

def analyze_with_retry(headline: str, client: OpenAI, retries: int = 2) -> dict:
    for _ in range(retries):
        try:
            return analyze_sentiment(headline, client)
        except Exception:
            continue
    return {
        "sentiment_score": 0,
        "key_risks": [],
        "summary": "Analysis failed."
    }


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
    score = analysis.get("sentiment_score", 0)
    css   = score_css_class(score)
    risks = analysis.get("key_risks", [])
    risks_css = score_css_class(score)
    risks_html = "".join(f'<div class="{risks_css}">▸ {r}</div>' for r in risks)

    content = item.get("content", {})
    title = content.get("title") if isinstance(content, dict) else None
    if not title:
        title = item.get("title", "No title available")

    # Sanitize title to prevent markdown parsing issues
    title = title.replace("`", "'").replace("<", "&lt;").replace(">", "&gt;")

    html = (
        '<div class="headline-card">'
        f'<div class="headline-text">#{idx} &nbsp;|&nbsp; {title}</div>'
        f'<div class="summary-text">{analysis.get("summary", "—")}</div>'
        f'<div style="font-size:0.75rem; margin-top:0.5rem; color:#8b949e;">'
        f'Sentiment: <span class="{css}">{score:+.2f}</span></div>'
        f'<div class="risk-list">{risks_html}</div>'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for the Streamlit dashboard."""

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
<div style="
    background:#161b22;
    padding:1rem 1.5rem;
    border:1px solid #21262d;
    border-radius:8px;
    margin-bottom:1.5rem;
">
    <div style="font-family:IBM Plex Mono; font-size:0.8rem; color:#ffffff;">
        EQUITY SENTIMENT TERMINAL
    </div>
    <div style="font-size:1.4rem; color:#ffffff; font-weight:600;">
        AI News Sentiment Dashboard
    </div>
</div>
""", unsafe_allow_html=True)

    # ── Session state init ───────────────────────────────────────────────────
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame()
    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data = {}

    # ── Input row ────────────────────────────────────────────────────────────
    ticker = st.text_input(
        "Ticker",
        placeholder="AAPL, MSFT, NVDA",
    )
    run = st.button("Run Analysis", use_container_width=True)

    # ── Analysis pipeline (only runs when button is pressed) ─────────────────
    if run:
        raw_input = ticker.upper().strip()
        if not raw_input:
            st.warning("Please enter a ticker symbol.")
            st.stop()

        tickers = [t.strip() for t in raw_input.split(",") if t.strip()]

        try:
            client = get_openai_client()
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

        # Clear previous results
        st.session_state.analysis_data = {}
        st.session_state.results_df = pd.DataFrame()

        def analyze_ticker(tick: str) -> tuple:
            """Fetch news and analyze sentiment for a single ticker."""
            try:
                news_items = fetch_news(tick)
            except ValueError as exc:
                return tick, None, str(exc)

            company  = get_company_name(tick)
            rows     = []
            analyses = []
            failures = 0

            for item in news_items:
                content = item.get("content", {})
                title = content.get("title") if isinstance(content, dict) else None
                if not title:
                    title = item.get("title", "No title available")
                try:
                    result = analyze_with_retry(title, client)
                except (ValueError, Exception):
                    failures += 1
                    result = {"sentiment_score": 0, "key_risks": [], "summary": "Analysis unavailable."}

                analyses.append(result)
                rows.append({
                    "ticker":          tick,
                    "company":         company,
                    "headline":        title,
                    "sentiment_score": result.get("sentiment_score", 0),
                    "sentiment_label": score_label(result.get("sentiment_score", 0)),
                    "summary":         result.get("summary", ""),
                    "key_risks":       " | ".join(result.get("key_risks", [])),
                    "raw_response":    json.dumps(result),
                })

            df = pd.DataFrame(rows)
            return tick, {
                "company":    company,
                "news_items": news_items,
                "analyses":   analyses,
                "df":         df,
                "failures":   failures,
            }, None

        with st.spinner(f"Analysing {len(tickers)} ticker(s) in parallel…"):
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(analyze_ticker, t): t for t in tickers}
                for future in concurrent.futures.as_completed(futures):
                    tick, data, error = future.result()
                    if error:
                        st.error(f"{tick}: {error}")
                        continue
                    st.session_state.analysis_data[tick] = data
                    st.session_state.results_df = pd.concat(
                        [st.session_state.results_df, data["df"]], ignore_index=True
                    )

    # ── Display (runs every rerun, reads from session state) ─────────────────
    if st.session_state.analysis_data:
        tickers = list(st.session_state.analysis_data.keys())
        tabs = st.tabs([f"📊 {t}" for t in tickers])

        for tab, tick in zip(tabs, tickers):
            data     = st.session_state.analysis_data[tick]
            company  = data["company"]
            news_items = data["news_items"]
            analyses = data["analyses"]
            df       = data["df"]
            failures = data["failures"]

            with tab:
                st.markdown(
                    f"<div class='ticker-badge'>{tick} · {company}</div>",
                    unsafe_allow_html=True,
                )

                # ── Price chart ──────────────────────────────────────────────
                st.markdown(
                    "<p style='font-family:IBM Plex Mono,monospace; font-size:0.75rem; "
                    "color:#8b949e; margin-bottom:0.25rem;'>PRICE HISTORY</p>",
                    unsafe_allow_html=True,
                )
                period = st.segmented_control(
                    f"Period_{tick}",
                    options=["1mo", "3mo", "6mo", "YTD", "1y", "2y", "5y", "Max"],
                    default="1mo",
                    label_visibility="collapsed",
                )
                hist = yf.Ticker(tick).history(period=period.lower())
                if not hist.empty:
                    st.line_chart(hist["Close"], color="#6001D2")

                st.markdown("<br>", unsafe_allow_html=True)
                st.caption(f"{len(news_items) - failures}/{len(news_items)} headlines successfully analysed")

                # ── Metrics ──────────────────────────────────────────────────
                avg_score = df["sentiment_score"].mean()
                max_score = df["sentiment_score"].max()
                min_score = df["sentiment_score"].min()
                std_dev   = df["sentiment_score"].std()

                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)

                def metric_card(col, label, value, css_cls):
                    arrow = "▲" if value > 0 else "▼" if value < 0 else "•"
                    with col:
                        st.markdown(
                            f"""<div class="metric-card">
                                <div class="label">{label}</div>
                                <div class="value {css_cls}">{arrow} {value:+.2f}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                metric_card(m1, "Avg Sentiment Score", avg_score, score_css_class(avg_score))
                metric_card(m2, "Highest Score",        max_score, score_css_class(max_score))
                metric_card(m3, "Lowest Score",         min_score, score_css_class(min_score))

                st.markdown("### Sentiment Distribution")
                st.bar_chart(df["sentiment_score"], color="#6001D2")

                st.markdown(
                    f"<p style='font-family:IBM Plex Mono,monospace; font-size:0.85rem; color:#6001D2;'>"
                    f"Sentiment dispersion (std dev): {std_dev:.2f}</p>",
                    unsafe_allow_html=True,
                )

                overall_label = score_label(avg_score)
                insight_color = "#3fb950" if avg_score > 0 else "#f85149" if avg_score < 0 else "#d29922"

                st.markdown(f"""
                <div style="
                    background:#161b22;
                    border-left:4px solid {insight_color};
                    padding:1rem;
                    margin-top:1rem;
                    border-radius:6px;
                ">
                    <b style="color:#ffffff;">Desk View:</b> <span style="color:#ffffff;">Current news flow suggests a <b>{overall_label}</b> bias,
                    with an average sentiment score of {avg_score:+.2f}.
                    Dispersion indicates {'strong consensus' if std_dev < 0.2 else 'mixed signals'}.</span>
                </div>
                """, unsafe_allow_html=True)

                overall_css = score_css_class(avg_score)
                st.markdown(
                    f"<p style='font-family:IBM Plex Mono,monospace; font-size:0.85rem; color:#6001D2;'>"
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

    # ── Export panel ─────────────────────────────────────────────────────────
    if not st.session_state.results_df.empty:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='font-size:1rem; letter-spacing:0.08em;'>EXPORT SESSION DATA</h3>",
            unsafe_allow_html=True,
        )

        with st.expander("View Raw Data"):
            st.dataframe(st.session_state.results_df, use_container_width=True)

        csv_bytes = st.session_state.results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇  Download CSV",
            data=csv_bytes,
            file_name="sentiment_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()