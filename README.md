# AI Financial Advisor — Render Deployment Guide

A full-stack AI-powered financial advisor app (Flask backend + HTML frontend) deployed as a single web service on Render.

---

## Project Structure

```
ai_fin_advisor/
├── app.py                  ← Main entry point (Render starts this)
├── requirements.txt        ← Python dependencies
├── render.yaml             ← Render auto-deploy config
├── .env.example            ← Copy to .env for local dev
├── frontend/
│   └── index.html          ← Full single-page frontend
└── backend/
    ├── app.py              ← Flask app factory
    ├── data.py             ← yFinance data fetching
    ├── database.py         ← SQLite persistence
    ├── ml_models.py        ← ML/AI models
    └── routes/
        ├── auth.py
        ├── market.py
        ├── portfolio.py
        ├── predict.py
        ├── risk.py
        ├── sentiment.py
        ├── backtest.py
        ├── screener.py
        └── chat.py
```

---

## Deploy to Render (Step-by-Step)

### Step 1 — Push to GitHub

```bash
cd ai_fin_advisor
git init
git add .
git commit -m "Initial commit — Render deploy"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ai-fin-advisor.git
git push -u origin main
```

### Step 2 — Create a Render Web Service

1. Go to [https://render.com](https://render.com) and sign in
2. Click **New → Web Service**
3. Connect your GitHub repo (`ai-fin-advisor`)
4. Fill in the settings:

| Setting | Value |
|---|---|
| **Name** | `ai-financial-advisor` |
| **Region** | Singapore (closest to India) |
| **Branch** | `main` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120` |
| **Instance Type** | Free (or Starter for better performance) |

### Step 3 — Set Environment Variables

In your Render service → **Environment** tab, add:

| Key | Value |
|---|---|
| `SECRET_KEY` | Click "Generate" or paste your own random string |
| `PYTHON_VERSION` | `3.11.0` |

> **Optional:** Add `ANTHROPIC_API_KEY` if you want real Claude-powered chat.

### Step 4 — Deploy

Click **Create Web Service**. Render will:
- Install dependencies (~3-5 minutes first time)
- Start your app with gunicorn
- Give you a URL like `https://ai-financial-advisor.onrender.com`

---

## Local Development

```bash
# Clone and set up
git clone https://github.com/YOUR_USERNAME/ai-fin-advisor.git
cd ai-fin-advisor

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit environment variables
cp .env.example .env

# Run locally
python app.py
# Open http://localhost:5000
```

---

## What's Different from the Replit Version

| Feature | Replit Version | Render Version |
|---|---|---|
| Entry point | `main.py` | `app.py` |
| WebSocket | flask-socketio + eventlet | Removed (not needed for free tier) |
| Server | eventlet | gunicorn (production-grade) |
| Requirements | Replit-specific packages | Clean, minimal |
| Config files | `.replit`, `replit.nix` | `render.yaml` |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/auth/login` | Login / auto-register |
| POST | `/api/auth/signup` | Sign up |
| GET | `/api/market/quotes` | Live stock quotes |
| GET/POST | `/api/portfolio/holdings` | Portfolio data |
| GET | `/api/portfolio/allocation` | Sector allocation |
| GET | `/api/portfolio/performance` | vs benchmark |
| GET | `/api/portfolio/rebalance` | AI rebalancing suggestions |
| POST | `/api/predict/forecast` | ML price forecast |
| GET | `/api/risk/analysis` | VaR, Sharpe, drawdown |
| GET | `/api/sentiment/analysis` | News sentiment |
| POST | `/api/backtest/run` | Strategy backtesting |
| GET | `/api/screener/scan` | Stock screener |
| POST | `/api/chat/send` | AI financial chat |

---

## Notes

- **SQLite database** (`backend/aifin.db`) is created automatically on first run.
- On Render's free tier, the service **spins down after 15 minutes of inactivity** — the first request after that may take ~30 seconds to wake up. Upgrade to Starter ($7/mo) to avoid this.
- **Persistent storage**: SQLite data is lost on redeploy on free tier. For production, use Render's Persistent Disk add-on or switch to PostgreSQL.
