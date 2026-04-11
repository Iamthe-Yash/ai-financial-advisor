"""routes/sentiment.py — TextBlob sentiment + simulated multi-LLM consensus"""
from flask import Blueprint, jsonify, request
from textblob import TextBlob
import re

sentiment_bp = Blueprint("sentiment", __name__, url_prefix="/api/sentiment")

# ── curated news feed (mimics the UI's NEWS array with real-looking data) ────
_NEWS = [
    {"t":"09:42","src":"ET Markets",    "txt":"RBI holds repo rate at 6.5%, signals optimism on inflation trajectory","sent":"BULLISH"},
    {"t":"09:31","src":"Reuters",       "txt":"Reliance Q4 profit surges 18% YoY, beats all analyst estimates","sent":"BULLISH"},
    {"t":"09:18","src":"Bloomberg",     "txt":"IT sector headwinds as US clients cut discretionary spend in H1","sent":"BEARISH"},
    {"t":"08:55","src":"CoinDesk",      "txt":"Bitcoin ETF records $1.2B net outflow on fresh regulatory fears","sent":"BEARISH"},
    {"t":"08:40","src":"Moneycontrol",  "txt":"Infosys raises FY26 revenue guidance to 4-7% on AI-driven deal wins","sent":"BULLISH"},
    {"t":"08:10","src":"CNBC",          "txt":"Global markets cautious ahead of tonight's FOMC minutes release","sent":"NEUTRAL"},
    {"t":"07:55","src":"LiveMint",      "txt":"Nifty Bank surges 1.4% as NIM expansion bets return post RBI hold","sent":"BULLISH"},
    {"t":"07:30","src":"Economic Times","txt":"FII flows turn positive after three-week selling streak in Indian equities","sent":"BULLISH"},
]

_BULLISH_WORDS = [
    "gain","surge","profit","bull","rise","growth","strong","beat","jump",
    "exceed","up","rally","record","upgrade","buy","breakout","positive",
    "optimism","recovery","green","high","outperform","boost","support",
]
_BEARISH_WORDS = [
    "fall","drop","loss","bear","decline","weak","crash","down","risk",
    "fear","miss","outflow","cut","sell","caution","headwind","negative",
    "concern","pressure","red","low","underperform","warning","correction",
]


@sentiment_bp.route("/news")
def news_feed():
    """Return live-ish news feed with pre-computed sentiment."""
    return jsonify(_NEWS)


@sentiment_bp.route("/sector")
def sector_sentiment():
    """Sector-level sentiment scores (0-100)."""
    sectors = [
        {"n":"US Tech",    "s":78},
        {"n":"Technology", "s":72},
        {"n":"Banking",    "s":65},
        {"n":"Pharma",     "s":54},
        {"n":"Crypto",     "s":34},
    ]
    return jsonify(sectors)


@sentiment_bp.route("/analyse", methods=["POST"])
def analyse():
    """
    Analyse arbitrary text with TextBlob (primary) + two
    scaled variants to simulate multi-LLM consensus.
    """
    body = request.get_json(silent=True) or {}
    text = body.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # ── TextBlob ──────────────────────────────────────────────────────────
    blob        = TextBlob(text)
    polarity    = float(blob.sentiment.polarity)        # -1 to +1
    subjectivity= float(blob.sentiment.subjectivity)    # 0 to 1

    # ── keyword boost ─────────────────────────────────────────────────────
    words = re.findall(r"\b\w+\b", text.lower())
    boost = 0.0
    for w in words:
        if w in _BULLISH_WORDS: boost += 0.12
        if w in _BEARISH_WORDS: boost -= 0.12
    combined = float(max(-1.0, min(1.0, polarity * 0.6 + boost * 0.4)))

    label = "BULLISH" if combined > 0.1 else "BEARISH" if combined < -0.1 else "NEUTRAL"

    # ── simulate GPT-4 / Gemini variants ─────────────────────────────────
    gpt_score  = round(combined * 0.93, 3)
    gem_score  = round(combined * 0.88, 3)
    gpt_label  = "BULLISH" if gpt_score > 0.1 else "BEARISH" if gpt_score < -0.1 else "NEUTRAL"
    gem_label  = "BULLISH" if gem_score > 0.1 else "BEARISH" if gem_score < -0.1 else "NEUTRAL"

    vader_compound = round(combined * 0.95, 3)

    return jsonify({
        "label":       label,
        "score":       round(combined, 3),
        "polarity":    round(polarity, 3),
        "subjectivity":round(subjectivity, 3),
        "vader_compound": vader_compound,
        "models": {
            "claude":  {"label": label,     "score": round(combined, 3)},
            "gpt4":    {"label": gpt_label, "score": gpt_score},
            "gemini":  {"label": gem_label, "score": gem_score},
        },
        "consensus": label,
    })
