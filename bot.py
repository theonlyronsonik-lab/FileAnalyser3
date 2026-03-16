import requests
import pandas as pd
import numpy as np
import os
import json
import asyncio
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta, timezone
from telegram import Bot
from telegram.error import TelegramError

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

API_KEY   = os.getenv("API_KEY", "")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID   = os.getenv("CHAT_ID", "")

SMTP_HOST   = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT   = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER   = os.getenv("SMTP_USER", "")
SMTP_PASS   = os.getenv("SMTP_PASS", "")
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")

SYMBOLS  = ["XAU/USD", "GBP/USD", "SPY", "QQQ"]
INTERVAL = "5min"

COOLDOWN_MINUTES = 15

RSI_OVERBOUGHT = 70
RSI_OVERSOLD   = 30

SIGNALS_FILE = "signals.json"

# State
last_signal_time = {}
signal_stack     = {}
active_trade     = {}
recent_signals   = []
trades_history   = []
symbol_state     = {}

# Nairobi timezone offset
NAIROBI_TZ = timezone(timedelta(hours=3))

SESSIONS = {
    "Asia":     (0,  10),
    "London":   (10,  15),
    "New York": (15, 22),
}

# ─────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────

def load_state():
    global recent_signals, trades_history
    if not os.path.exists(SIGNALS_FILE):
        return
    try:
        with open(SIGNALS_FILE) as f:
            data = json.load(f)
        recent_signals = data.get("recent_signals", [])
        trades_history = data.get("trades_history", [])
        print(f"Loaded {len(trades_history)} historical trades, {len(recent_signals)} recent signals")
    except Exception as e:
        print(f"State load error: {e}")


def compute_stats():
    closed = [t for t in trades_history if t.get("outcome") in ("WIN", "LOSS")]
    wins   = [t for t in closed if t["outcome"] == "WIN"]

    by_asset   = {}
    by_session = {}

    for sym in SYMBOLS:
        sym_trades = [t for t in closed if t["symbol"] == sym]
        sym_wins   = [t for t in sym_trades if t["outcome"] == "WIN"]
        by_asset[sym] = {
            "total":    len(sym_trades),
            "wins":     len(sym_wins),
            "losses":   len(sym_trades) - len(sym_wins),
            "win_rate": round(len(sym_wins) / len(sym_trades) * 100, 1) if sym_trades else 0,
        }

    for sess in list(SESSIONS.keys()) + ["Off-Hours"]:
        sess_trades = [t for t in closed if sess in t.get("session", "")]
        sess_wins   = [t for t in sess_trades if t["outcome"] == "WIN"]
        by_session[sess] = {
            "total":    len(sess_trades),
            "wins":     len(sess_wins),
            "losses":   len(sess_trades) - len(sess_wins),
            "win_rate": round(len(sess_wins) / len(sess_trades) * 100, 1) if sess_trades else 0,
        }

    total = len(closed)
    return {
        "total":      total,
        "wins":       len(wins),
        "losses":     total - len(wins),
        "pending":    len([t for t in trades_history if t.get("outcome") == "OPEN"]),
        "win_rate":   round(len(wins) / total * 100, 1) if total else 0,
        "by_asset":   by_asset,
        "by_session": by_session,
    }


def save_state(session_on, current_sessions):
    data = {
        "bot_status":       "running",
        "last_scan":        datetime.now(NAIROBI_TZ).strftime("%Y-%m-%d %H:%M:%S Nairobi"),
        "session_active":   session_on,
        "current_sessions": current_sessions,
        "symbols":          {},
        "recent_signals":   recent_signals[-50:],
        "trades_history":   trades_history[-200:],
        "stats":            compute_stats(),
    }
    for sym in SYMBOLS:
        st = symbol_state.get(sym, {})
        data["symbols"][sym] = {
            "price":        st.get("price"),
            "rsi":          st.get("rsi"),
            "sma200":       st.get("sma200"),
            "atr":          st.get("atr"),
            "trend":        st.get("trend"),
            "active_trade": active_trade.get(sym),
            "last_signal":  next(
                (s for s in reversed(recent_signals) if s["symbol"] == sym), None
            ),
        }
    try:
        with open(SIGNALS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Save error: {e}")


def init_state():
    data = {
        "bot_status": "starting",
        "last_scan": None,
        "session_active": False,
        "current_sessions": [],
        "symbols": {sym: {"price": None, "rsi": None, "sma200": None,
                          "atr": None, "trend": None,
                          "active_trade": None, "last_signal": None}
                    for sym in SYMBOLS},
        "recent_signals": recent_signals,
        "trades_history": trades_history,
        "stats": compute_stats(),
    }
    with open(SIGNALS_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def get_data(symbol):
    url = (f"https://api.twelvedata.com/time_series"
           f"?symbol={symbol}&interval={INTERVAL}&outputsize=210&apikey={API_KEY}")
    try:
        r = requests.get(url, timeout=15).json()
    except Exception as e:
        print(f"Fetch error {symbol}: {e}")
        return None

    if "values" not in r:
        print(f"No data for {symbol}: {r.get('message', '')}")
        return None

    df = pd.DataFrame(r["values"]).iloc[::-1].reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df

def calc_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_sma200(series):
    return series.rolling(200).mean()

def calc_atr(df, period=14):
    h  = df["high"]
    l  = df["low"]
    c  = df["close"]
    tr = pd.concat([
        (h - l),
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def pivot_low(series, left=5, right=5):
    pivots = []
    vals   = series.values
    for i in range(left, len(vals) - right):
        window = vals[i - left: i + right + 1]
        if vals[i] == np.min(window):
            pivots.append(i)
    return pivots

def pivot_high(series, left=5, right=5):
    pivots = []
    vals   = series.values
    for i in range(left, len(vals) - right):
        window = vals[i - left: i + right + 1]
        if vals[i] == np.max(window):
            pivots.append(i)
    return pivots

def bullish_div(df):
    lows = pivot_low(df["low"])
    if len(lows) < 2:
        return False, None
    i1, i2 = lows[-2], lows[-1]
    price_ll = df["low"].iloc[i2] < df["low"].iloc[i1]
    rsi_hl   = df["rsi"].iloc[i2] > df["rsi"].iloc[i1]
    if price_ll and rsi_hl:
        return True, i2
    return False, None

def bearish_div(df):
    highs = pivot_high(df["high"])
    if len(highs) < 2:
        return False, None
    i1, i2 = highs[-2], highs[-1]
    price_hh = df["high"].iloc[i2] > df["high"].iloc[i1]
    rsi_lh   = df["rsi"].iloc[i2]  < df["rsi"].iloc[i1]
    if price_hh and rsi_lh:
        return True, i2
    return False, None

# ─────────────────────────────────────────────
# Remaining functions like double_confirm, open_trade_record, close_trade_record,
# check_rsi_tp_zone, check_tp, send_telegram, send_email, get_market_context, is_high_quality
# remain unchanged; main() will use Nairobi TZ instead of UTC.
# ─────────────────────────────────────────────

# In main(), replace:
# datetime.now(timezone.utc) → datetime.now(NAIROBI_TZ)
# And update session logic accordingly.

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

async def main():
    load_state()
    init_state()
    print(f"Bot started | Symbols: {SYMBOLS}")

    await send_telegram(
        f"🤖 Signal Bot Online\n"
        f"Symbols: {', '.join(SYMBOLS)}\n"
        f"Interval: {INTERVAL}\n"
        f"SL: Divergence candle wick\n"
        f"TP1: RSI overbought/oversold alert\n"
        f"TP2: Opposite double signal"
    )

    while True:
        try:
            now = datetime.now(NAIROBI_TZ)
            hour = now.hour

            sessions = get_active_sessions()  # This function can also use Nairobi TZ
            sess_on  = sessions != ["Off-Hours"]
            sess_str = session_label(sessions)

            if not sess_on:
                now_str = now.strftime("%H:%M")
                print(f"[{now_str} Nairobi] Off-hours, sleeping 60s…")
                save_state(False, sessions)
                await asyncio.sleep(60)
                continue

            for symbol in SYMBOLS:
                df = get_data(symbol)
                if df is None:
                    continue

                df["rsi"]    = calc_rsi(df["close"])
                df["sma200"] = calc_sma200(df["close"])
                df["atr"]    = calc_atr(df)

                price   = round(df["close"].iloc[-1], 5)
                rsi     = round(df["rsi"].iloc[-1], 2)
                sma200  = df["sma200"].iloc[-1]
                atr     = df["atr"].iloc[-1]

                sma200_val = round(sma200, 5) if not pd.isna(sma200) else None
                atr_val    = round(atr,    5) if not pd.isna(atr)    else None
                trend      = None
                if sma200_val:
                    trend = "BULLISH" if price > sma200_val else "BEARISH"

                symbol_state[symbol] = {
                    "price":  price,
                    "rsi":    rsi,
                    "sma200": sma200_val,
                    "atr":    atr_val,
                    "trend":  trend,
                }

                # BUY / SELL handling remains as in your current code
                # Only now using Nairobi time for timestamps
                # Include double confirmation, divergence detection, RSI TP, email/telegram

            save_state(sess_on, sessions)
            await asyncio.sleep(60)

        except Exception as e:
            print(f"Main loop error: {e}")
            await asyncio.sleep(60)

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(main())
