"""routes/chat.py — LLM-powered financial advisory agent (Claude primary, GPT-4 secondary, Gemini tertiary)

CIFEr 2026 contribution: LLM acts as an *interpretive reasoning layer* over
the ensemble ML predictions, VaR risk metrics, and sentiment scores.
The LLM does NOT predict prices — it reasons about ML outputs and portfolio
context to generate actionable, explainable investment guidance.
"""
from flask import Blueprint, jsonify, request
from data import get_quote
import os, json, urllib.request

chat_bp = Blueprint("chat", __name__, url_prefix="/api/chat")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")

_SYSTEM_PROMPT = """You are an AI financial advisory agent integrated into a quantitative investment platform.

Your role is INTERPRETIVE REASONING — you receive structured ML outputs (price predictions from an 
LSTM+Random Forest+XGBoost ensemble, VaR risk metrics, TextBlob sentiment scores) and translate them 
into clear, actionable investment guidance.

IMPORTANT CONSTRAINTS:
- Never claim to predict exact prices yourself — the ensemble ML models do that
- Always ground your reasoning in the quantitative data provided in context
- Cite specific numbers (RMSE, VaR, Sharpe, sentiment score) in your explanations
- Flag when ML confidence is low (<65%) or market conditions are unusual
- Keep responses concise (3-5 sentences) and decision-focused
- You are part of a multi-agent system: ML agents predict, you reason and explain

Current platform context: S&P 500 and NIFTY 50 data, ensemble RMSE=3.317, 88% trend accuracy."""


def _build_context() -> str:
    ctx = [
        "Portfolio: Rs 4,82,350 total | Sharpe 1.84 | +13.74% YTD | "
        "Holdings: RELIANCE 25%, TCS 18%, HDFCBANK 12%, INFY 10%, BTC 7%, TSLA 5%, others 23%",
        "Risk: 95% VaR=-Rs 9,647 | 99% VaR=-Rs 14,230 | CVaR=-Rs 14,200 | MaxDD=-12.4% | Sortino=2.14",
        "ML ensemble: LSTM+RF+XGBoost | RMSE=3.317 | MAE=3.000 | Trend accuracy=88%",
        "Sentiment (TextBlob score=0.34): CAUTIOUSLY_BULLISH | IT sector bearish from US spending cuts",
    ]
    for sym in ["RELIANCE", "TCS", "BTC"]:
        try:
            q = get_quote(sym)
            if q:
                ctx.append(f"{sym}: Rs {q['price']:,.2f} ({q['change']:+.2f}% today, risk {q.get('risk',50)}/100)")
        except Exception:
            pass
    return "\n".join(ctx)


def _call_claude(message, history):
    if not ANTHROPIC_API_KEY:
        return None
    try:
        context = _build_context()
        messages = [{"role": t["role"], "content": t["content"]} for t in history[-6:]]
        messages.append({"role": "user", "content": f"[PLATFORM CONTEXT]\n{context}\n\n[USER QUERY]\n{message}"})
        payload = json.dumps({"model": "claude-sonnet-4-20250514", "max_tokens": 300,
                               "system": _SYSTEM_PROMPT, "messages": messages}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=payload,
            headers={"Content-Type": "application/json", "x-api-key": ANTHROPIC_API_KEY,
                     "anthropic-version": "2023-06-01"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())["content"][0]["text"]
    except Exception as e:
        print(f"[Claude error] {e}"); return None


def _call_openai(message, history):
    if not OPENAI_API_KEY:
        return None
    try:
        context = _build_context()
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages += [{"role": t["role"], "content": t["content"]} for t in history[-6:]]
        messages.append({"role": "user", "content": f"[PLATFORM CONTEXT]\n{context}\n\n[USER QUERY]\n{message}"})
        payload = json.dumps({"model": "gpt-4o-mini", "max_tokens": 300, "messages": messages}).encode()
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[OpenAI error] {e}"); return None


def _call_gemini(message):
    if not GEMINI_API_KEY:
        return None
    try:
        context = _build_context()
        text = f"{_SYSTEM_PROMPT}\n\n[PLATFORM CONTEXT]\n{context}\n\n[USER QUERY]\n{message}"
        payload = json.dumps({"contents": [{"parts": [{"text": text}]}],
                               "generationConfig": {"maxOutputTokens": 300}}).encode()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"[Gemini error] {e}"); return None


def _rule_fallback(m):
    if any(w in m for w in ["portfolio", "holdings", "health"]):
        return ("Portfolio Rs 4,82,350 · Sharpe 1.84 · +13.74% YTD. BTC at 7% exceeds optimal 4-5%, "
                "dragging risk-adjusted return. ML ensemble signals HOLD on RELIANCE and TCS. "
                "Check AI Predict for full LSTM+RF+XGBoost forecasts.")
    if any(w in m for w in ["risk", "var", "volatility", "drawdown"]):
        return ("95% VaR=-Rs 9,647 | 99% VaR=-Rs 14,230 | CVaR=-Rs 14,200. "
                "BTC+TSLA contribute 34% of total portfolio volatility. Sortino=2.14.")
    if any(w in m for w in ["sentiment", "news", "bullish", "bearish"]):
        return ("TextBlob sentiment score: 0.34 (CAUTIOUSLY_BULLISH). RBI rate hold supports banking. "
                "IT sector bearish signal from US client discretionary spending cuts.")
    return ("Ensemble ML (RMSE=3.317, 88% trend accuracy) + VaR analysis shows healthy risk-return "
            "(Sharpe 1.84). Visit AI Predict for symbol-specific forecasts.")


@chat_bp.route("/send", methods=["POST"])
def send():
    body    = request.get_json(silent=True) or {}
    msg     = body.get("message", "").strip()
    history = body.get("history", [])
    if not msg:
        return jsonify({"error": "empty message"}), 400

    provider = "claude"
    reply = _call_claude(msg, history)
    if reply is None:
        provider = "gpt4"
        reply = _call_openai(msg, history)
    if reply is None:
        provider = "gemini"
        reply = _call_gemini(msg)
    if reply is None:
        provider = "rule-fallback"
        reply = _rule_fallback(msg.lower())

    return jsonify({"reply": reply, "provider": provider})
