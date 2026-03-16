# RonFX Gold Signals

## Overview

Python trading signal bot monitoring 4 assets (XAU/USD, GBP/USD, SPY, QQQ) with RSI divergence + SMA200 trend filter, Telegram/email alerts, and a modern live dashboard.

## Architecture

- **bot.py** — async trading bot (5-min loop), writes all state to `signals.json`
- **app.py** — Flask web dashboard (port 5000), reads `signals.json` via `/api/data`
- **start.sh** — launches both: `bot.py` in background, `app.py` in foreground
- **templates/index.html** — dark-themed modern dashboard (4 pages, Chart.js)

## Assets Monitored

| Symbol  | Notes            |
|---------|-----------------|
| XAU/USD | Gold vs USD     |
| GBP/USD | Cable           |
| SPY     | S&P 500 ETF     |
| QQQ     | NASDAQ 100 ETF  |

## Trading Strategy

1. **RSI Divergence** — pivot-based true divergence detection
2. **Double Confirmation** — two consecutive divergence signals required
3. **SMA200 Trend Filter** — labels signals as Trend Aligned or Counter-Trend
4. **Stop Loss** — at the wick of the divergence candle (pivot low/high)
5. **Take Profit Model 1** — RSI reaches overbought (>70 for BUY) or oversold (<30 for SELL) → Telegram alert sent so user can decide
6. **Take Profit Model 2** — Opposite double confirmation signal fires → trade closed as WIN
7. **Cooldown** — 15 minutes between signals per symbol

## Market Context

Each signal includes a brief rule-based market context tip covering RSI state, SMA200 trend, ATR volatility, and session timing.

## Sessions (UTC)

- Asia: 02:00–10:00
- London: 07:00–16:00
- New York: 13:00–22:00

## Alert System

- **Telegram** — all signals + RSI TP zone alerts
- **Email (SMTP)** — HIGH QUALITY only: trend-aligned + 14:00–20:00 UTC

## Dashboard Pages

| Page      | Content                                          |
|-----------|--------------------------------------------------|
| Home      | Bot status, stat cards, live symbol cards        |
| Signals   | All signals with entry, SL, RSI, alignment       |
| Trades    | Historical trades with outcome tracking          |
| Analytics | Chart.js charts, win rate bars by asset/session  |

## Environment Variables

| Key         | Purpose                    |
|-------------|---------------------------|
| API_KEY     | Twelve Data API key        |
| BOT_TOKEN   | Telegram bot token         |
| CHAT_ID     | Telegram chat ID           |
| SMTP_USER   | Email sender               |
| SMTP_PASS   | Email app password         |
| ALERT_EMAIL | Email recipient            |
| SMTP_HOST   | SMTP host (gmail default)  |
| SMTP_PORT   | SMTP port (587 default)    |

## Stack

- Python 3.12
- Flask 3 (dashboard)
- python-telegram-bot 21.9
- pandas, numpy (indicators + data)
- Chart.js (frontend charts, CDN)

## Workspace (TypeScript Monorepo)

- `artifacts/api-server/` — Express API server (unused for this project)
- `artifacts/mockup-sandbox/` — UI component sandbox
- `lib/` — shared TypeScript libraries
