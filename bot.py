import requests
import pandas as pd
import numpy as np
import os
import json
import asyncio
import smtplib
import time
from email.mime.text import MIMEText
from datetime import datetime, timedelta, timezone
from telegram import Bot
from telegram.error import TelegramError

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

API_KEY = os.getenv("API_KEY", "")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID   = os.getenv("CHAT_ID", "")

SMTP_HOST   = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT   = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER   = os.getenv("SMTP_USER", "")
SMTP_PASS   = os.getenv("SMTP_PASS", "")
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")

SYMBOLS  = ["XAU/USD", "EUR/USD", "AUD/CAD", "CAD/JPY", "EUR/JPY"]
INTERVAL = "5min"

COOLDOWN_MINUTES = 15

RSI_OVERBOUGHT = 70
RSI_OVERSOLD   = 30

DATA_DIR = "/app/data"
os.makedirs(DATA_DIR, exist_ok=True)
SIGNALS_FILE = os.path.join(DATA_DIR, "signals.json")

# Symbol-specific SL buffers (applied beyond the 5-candle wick)
SL_BUFFERS = {
    "XAU/USD": 0.50,
    "EUR/USD": 0.0003,
    "AUD/CAD":     0.10,
    "CAD/JPY":     0.10,
    "EUR/JPY":     0.10,
}

# Pip sizes per symbol
PIP_SIZES = {
    "XAU/USD": 0.1,
    "EUR/USD": 0.0001,
    "AUD/CAD":     0.01,
    "CAD/JPY":     0.01,
    "EUR/JPY":     0.01,
}

LOT_SIZE = 0.01  # Default lot size

# State
last_signal_time = {}
signal_stack     = {}
active_trade     = {}
recent_signals   = []
trades_history   = []
symbol_state     = {}
last_div_time    = {}   # {symbol: {"BULL": candle_dt_str, "BEAR": candle_dt_str}}

SESSIONS = {
    "Asia":     (0,  7),
    "London":   (7,  15),
    "New York": (14, 20),
}


# ─────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────

def load_state():
    global recent_signals, trades_history, active_trade
    if not os.path.exists(SIGNALS_FILE):
        return
    try:
        with open(SIGNALS_FILE) as f:
            data = json.load(f)
        recent_signals = data.get("recent_signals", [])
        trades_history = data.get("trades_history", [])
        print(f"Loaded {len(trades_history)} historical trades, {len(recent_signals)} recent signals")

        # Restore active trades from the most recent OPEN record per symbol
        for sym in SYMBOLS:
            open_trades = [t for t in trades_history
                           if t["symbol"] == sym and t.get("outcome") == "OPEN"]
            if open_trades:
                last = open_trades[-1]
                active_trade[sym] = {
                    "type":          last["type"],
                    "entry":         last["entry"],
                    "sl":            last["sl"],
                    "pip_size":      PIP_SIZES.get(sym, 0.0001),
                    "trend_aligned": last.get("trend_aligned", False),
                    "label":         last.get("label", ""),
                    "session":       last.get("session", ""),
                    "rsi_alerted":   False,
                }
        if active_trade:
            print(f"Restored active trades: {list(active_trade.keys())}")
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
        "last_scan":        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "session_active":   session_on,
        "current_sessions": current_sessions,
        "symbols":          {},
        "recent_signals":   recent_signals[-50:],
        "trades_history":   trades_history[-200:],
        "stats":            compute_stats(),
        "pip_sizes":        PIP_SIZES,
        "lot_size":         LOT_SIZE,
    }
    for sym in SYMBOLS:
        st = symbol_state.get(sym, {})
        current_price = st.get("price")
        at = active_trade.get(sym)
        # Attach live pips/profit to active trade for dashboard display
        if at and current_price is not None:
            live_pips   = calc_pips(sym, at["entry"], current_price, at["type"])
            live_profit = calc_profit(live_pips)
            at_data = {**at, "current_price": current_price,
                       "current_pips": live_pips, "current_profit": live_profit}
        else:
            at_data = at
        data["symbols"][sym] = {
            "price":        current_price,
            "rsi":          st.get("rsi"),
            "sma200":       st.get("sma200"),
            "atr":          st.get("atr"),
            "trend":        st.get("trend"),
            "active_trade": at_data,
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
# SESSIONS
# ─────────────────────────────────────────────

def get_active_sessions():
    hour = datetime.now(timezone.utc).hour
    active = [name for name, (s, e) in SESSIONS.items() if s <= hour <= e]
    return active if active else ["Off-Hours"]


def session_active():
    return get_active_sessions() != ["Off-Hours"]


def session_label(sessions):
    return " / ".join(sessions) if sessions else "Off-Hours"


# ─────────────────────────────────────────────
# MARKET CONTEXT (rule-based tips)
# ─────────────────────────────────────────────

def get_market_context(symbol, price, rsi, sma200, atr, trend):
    tips = []

    if rsi is None:
        return "Insufficient data."

    if rsi >= RSI_OVERBOUGHT:
        tips.append(f"RSI {rsi:.1f} — overbought, momentum may be exhausting")
    elif rsi >= 70:
        tips.append(f"RSI {rsi:.1f} — elevated, strong momentum but watch for pullback")
    elif rsi <= RSI_OVERSOLD:
        tips.append(f"RSI {rsi:.1f} — oversold, potential bounce zone")
    elif rsi <= 30:
        tips.append(f"RSI {rsi:.1f} — weak, selling pressure present")
    else:
        tips.append(f"RSI {rsi:.1f} — neutral zone")

    if trend == "BULLISH":
        tips.append("Above SMA200 — long-term uptrend")
    elif trend == "BEARISH":
        tips.append("Below SMA200 — long-term downtrend")

    if atr and price:
        vol_pct = (atr / price) * 100
        if vol_pct > 1.0:
            tips.append("High volatility — consider reduced size")
        elif vol_pct < 0.2:
            tips.append("Low volatility — tight conditions")

    hour = datetime.now(timezone.utc).hour
    if 14 <= hour <= 20:
        tips.append("NY session active — peak liquidity window")
    elif 7 <= hour <= 10:
        tips.append("London/Asia overlap — elevated volatility possible")

    return " | ".join(tips)


# ─────────────────────────────────────────────
# ALERTS
# ─────────────────────────────────────────────

async def send_telegram(msg):
    if not BOT_TOKEN:
        print(msg)
        return
    try:
        bot = Bot(token=BOT_TOKEN)
        await bot.send_message(chat_id=CHAT_ID, text=msg)
    except TelegramError as e:
        print(f"Telegram error: {e}")


def send_email(subject, body):
    if not (SMTP_USER and SMTP_PASS and ALERT_EMAIL):
        return
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = SMTP_USER
        msg["To"]      = ALERT_EMAIL
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.ehlo()
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(SMTP_USER, ALERT_EMAIL, msg.as_string())
        print("Email alert sent")
    except Exception as e:
        print(f"Email error: {e}")


def is_high_quality(trend_aligned):
    hour = datetime.now(timezone.utc).hour
    return trend_aligned and (7 <= hour <= 20)


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def get_data(symbol):
    url = (f"https://api.twelvedata.com/time_series"
           f"?symbol={symbol}&interval={INTERVAL}&outputsize=210&apikey={API_KEY}")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=(10, 30))
            r = resp.json()
        except requests.exceptions.Timeout as e:
            print(f"[{symbol}] Timeout on attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
                continue
            print(f"[{symbol}] All {max_retries} attempts timed out, skipping symbol")
            return None
        except ValueError as e:
            # JSON decode error — API returned empty/invalid body
            print(f"[{symbol}] JSON parse error on attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
                continue
            print(f"[{symbol}] All {max_retries} attempts returned invalid JSON, skipping symbol")
            return None
        except requests.exceptions.RequestException as e:
            print(f"[{symbol}] Network error on attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
                continue
            print(f"[{symbol}] All {max_retries} attempts failed with network error, skipping symbol")
            return None

        if "values" not in r:
            # API-level error (bad key, rate limit, unknown symbol, etc.) — no point retrying
            print(f"[{symbol}] API error (no values): {r.get('message', r.get('status', 'unknown error'))}")
            return None

        df = pd.DataFrame(r["values"]).iloc[::-1].reset_index(drop=True)
        for c in ["open", "high", "low", "close"]:
            df[c] = df[c].astype(float)

        # ── Data validation ──────────────────────────────────────────────
        if len(df) < 200:
            print(f"[{symbol}] Insufficient candles: got {len(df)}, need at least 200 — skipping")
            return None

        ohlc_cols = ["open", "high", "low", "close"]
        if df[ohlc_cols].isnull().any().any():
            print(f"[{symbol}] OHLC data contains NaN values — skipping")
            return None

        if (df[ohlc_cols] == 0).any().any():
            print(f"[{symbol}] OHLC data contains zero values — skipping")
            return None

        return df

    return None


# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# PIVOTS
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# SL / PIPS / PROFIT HELPERS
# ─────────────────────────────────────────────

def get_sl_buy(df, symbol):
    """SL = lowest low of last 5 candles minus buffer."""
    buf  = SL_BUFFERS.get(symbol, 0.0001)
    low5 = df["low"].iloc[-5:].min()
    return round(low5 - buf, 5)


def get_sl_sell(df, symbol):
    """SL = highest high of last 5 candles plus buffer."""
    buf   = SL_BUFFERS.get(symbol, 0.0001)
    high5 = df["high"].iloc[-5:].max()
    return round(high5 + buf, 5)


def calc_pips(symbol, entry, close_price, direction):
    pip  = PIP_SIZES.get(symbol, 0.0001)
    diff = (close_price - entry) if direction == "BUY" else (entry - close_price)
    return round(diff / pip, 1)


def calc_profit(pips, lot_size=None):
    ls = lot_size if lot_size is not None else LOT_SIZE
    return round(pips * ls * 10, 2)


# ─────────────────────────────────────────────
# DIVERGENCE
# ─────────────────────────────────────────────

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
# DOUBLE CONFIRMATION
# ─────────────────────────────────────────────

def double_confirm(symbol, signal):
    if symbol not in signal_stack:
        signal_stack[symbol] = []
    signal_stack[symbol].append(signal)
    if len(signal_stack[symbol]) > 2:
        signal_stack[symbol].pop(0)
    if signal_stack[symbol] == ["BUY", "BUY"]:
        return "BUY"
    if signal_stack[symbol] == ["SELL", "SELL"]:
        return "SELL"
    return None


# ─────────────────────────────────────────────
# TRADE RECORDS
# ─────────────────────────────────────────────

def open_trade_record(symbol, sig_type, entry, sl, trend_aligned, label, sess):
    rec = {
        "symbol":        symbol,
        "type":          sig_type,
        "time":          datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "close_time":    None,
        "entry":         entry,
        "sl":            sl,
        "outcome":       "OPEN",
        "trend_aligned": trend_aligned,
        "label":         label,
        "session":       sess,
    }
    trades_history.append(rec)
    return rec


def close_trade_record(symbol, outcome, close_price=None):
    for t in reversed(trades_history):
        if t["symbol"] == symbol and t["outcome"] == "OPEN":
            t["outcome"]    = outcome
            t["close_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            if close_price is not None:
                raw_pips = calc_pips(symbol, t["entry"], close_price, t["type"])
                if outcome == "LOSS":
                    raw_pips = -abs(raw_pips)
                t["close_price"] = close_price
                t["pips"]        = raw_pips
                t["profit"]      = calc_profit(raw_pips)
            print(f"Trade closed: {symbol} → {outcome} | pips: {t.get('pips','?')} | profit: ${t.get('profit','?')}")
            return t
    return None


# ─────────────────────────────────────────────
# RSI TAKE PROFIT ZONE CHECK (TP Model 1)
# Alert user when RSI reaches overbought/oversold
# ─────────────────────────────────────────────

async def check_rsi_tp_zone(symbol, rsi):
    if symbol not in active_trade:
        return

    trade = active_trade[symbol]

    if trade["type"] == "BUY" and rsi >= RSI_OVERBOUGHT:
        if not trade.get("rsi_alerted"):
            msg = (
                f"⚠️ RSI TP ZONE — {symbol}\n"
                f"BUY trade @ {trade['entry']} | RSI: {rsi:.1f} (OVERBOUGHT)\n"
                f"Consider closing for profit or hold for opposite signal.\n"
                f"Session: {trade.get('session', 'N/A')}"
            )
            print(msg)
            await send_telegram(msg)
            active_trade[symbol]["rsi_alerted"] = True

    elif trade["type"] == "SELL" and rsi <= RSI_OVERSOLD:
        if not trade.get("rsi_alerted"):
            msg = (
                f"⚠️ RSI TP ZONE — {symbol}\n"
                f"SELL trade @ {trade['entry']} | RSI: {rsi:.1f} (OVERSOLD)\n"
                f"Consider closing for profit or hold for opposite signal.\n"
                f"Session: {trade.get('session', 'N/A')}"
            )
            print(msg)
            await send_telegram(msg)
            active_trade[symbol]["rsi_alerted"] = True

    # Reset alert flag if RSI moves back out of the zone
    elif trade.get("rsi_alerted"):
        if trade["type"] == "BUY" and rsi < RSI_OVERBOUGHT - 5:
            active_trade[symbol]["rsi_alerted"] = False
        elif trade["type"] == "SELL" and rsi > RSI_OVERSOLD + 5:
            active_trade[symbol]["rsi_alerted"] = False


# ─────────────────────────────────────────────
# SL CHECK — Price hits stop loss
# ─────────────────────────────────────────────

async def check_sl(symbol, price):
    if symbol not in active_trade:
        return
    trade = active_trade[symbol]
    sl = trade.get("sl")
    if sl is None:
        return
    hit = (trade["type"] == "BUY" and price <= sl) or \
          (trade["type"] == "SELL" and price >= sl)
    if not hit:
        return
    raw_pips = calc_pips(symbol, trade["entry"], price, trade["type"])
    raw_pips = -abs(raw_pips)
    profit   = calc_profit(raw_pips)
    msg = (
        f"🛑 SL HIT for LTF signal— {symbol}\n"
        f"{trade['type']} @ {trade['entry']} | SL: {sl}\n"
        f"Close: {price} | Pips: {raw_pips} | P&L: ${profit}\n"
        f"Session: {trade.get('session', 'N/A')}"
    )
    print(msg)
    await send_telegram(msg)
    close_trade_record(symbol, "LOSS", close_price=price)
    del active_trade[symbol]


# ─────────────────────────────────────────────
# TP CHECK — Opposite Signal (TP Model 2)
# ─────────────────────────────────────────────

async def check_tp(symbol, signal, price=None):
    if symbol not in active_trade:
        return

    trade = active_trade[symbol]
    closed_type = None

    if signal == "BUY" and trade["type"] == "SELL":
        closed_type = "SELL"
    elif signal == "SELL" and trade["type"] == "BUY":
        closed_type = "BUY"

    if not closed_type:
        return

    raw_pips = calc_pips(symbol, trade["entry"], price, trade["type"]) if price else 0
    profit   = calc_profit(raw_pips)
    msg = (
        f"✅ TP HIT for LTF signal (Opposite Signal) — {symbol}\n"
        f"{closed_type} @ {trade['entry']} → Close: {price}\n"
        f"Pips: {raw_pips} | P&L: ${profit} | Outcome: WIN"
    )
    print(msg)
    await send_telegram(msg)
    close_trade_record(symbol, "WIN", close_price=price)
    del active_trade[symbol]


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

async def main():
    load_state()
    init_state()
    print(f"Bot started | Symbols: {SYMBOLS}")

    await send_telegram(
        f"🤖 LTF Signal Bot Online\n"
        f"Symbols: {', '.join(SYMBOLS)}\n"
        f"Interval: {INTERVAL}\n"
        f"SL: Divergence candle wick\n"
        f"TP1: RSI overbought/oversold alert\n"
        f"TP2: Opposite double signal"
    )

    while True:
        try:
            sessions = get_active_sessions()
            sess_on  = sessions != ["Off-Hours"]
            sess_str = session_label(sessions)

            if not sess_on:
                now_str = datetime.now(timezone.utc).strftime("%H:%M")
                print(f"[{now_str} UTC] Off-hours, sleeping 60s…")
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

                price   = round(df["close"].iloc[-2], 5)
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

                # Check RSI TP zone for active trade (TP Model 1)
                await check_rsi_tp_zone(symbol, rsi)

                # Check if SL has been hit by current price
                await check_sl(symbol, price)

                bull, bull_idx = bullish_div(df)
                bear, bear_idx = bearish_div(df)

                now = datetime.now(timezone.utc)

                if bull and bull_idx is not None:
                    print(f"[{symbol}] Bullish divergence detected at candle index {bull_idx} | RSI: {rsi:.2f} | Price: {price}")
                if bear and bear_idx is not None:
                    print(f"[{symbol}] Bearish divergence detected at candle index {bear_idx} | RSI: {rsi:.2f} | Price: {price}")

                # If there's already an active trade, skip new signal generation
                if symbol in active_trade:
                    continue

                # ── BUY ──
                if bull and bull_idx is not None:
                    # Only signal if this is a NEW divergence candle (not same one as last signal)
                    bull_candle_dt = str(df["datetime"].iloc[bull_idx]) if "datetime" in df.columns else str(bull_idx)
                    sym_div = last_div_time.setdefault(symbol, {})
                    if sym_div.get("BULL") == bull_candle_dt:
                        bull = False  # same candle, skip
                        print(f"[{symbol}] Bullish divergence already signalled for candle {bull_candle_dt}, skipping")
                    else:
                        await check_tp(symbol, "BUY", price)

                if bull and bull_idx is not None:
                    ds = double_confirm(symbol, "BUY")
                    print(f"[{symbol}] BUY double-confirm stack: {signal_stack.get(symbol, [])} → result: {ds}")

                    if ds == "BUY":
                        entry         = price
                        sl            = get_sl_buy(df, symbol)
                        pip_size      = PIP_SIZES.get(symbol, 0.0001)
                        trend_aligned = (trend == "BULLISH")
                        label         = "Trend Aligned Signal" if trend_aligned else "Counter-Trend Signal"
                        ts            = now.strftime("%Y-%m-%d %H:%M UTC")
                        context       = get_market_context(symbol, price, rsi, sma200_val, atr_val, trend)
                        risk_pips     = round((entry - sl) / pip_size, 1)

                        print(f"[{symbol}] Opening BUY trade | Entry: {entry} | SL: {sl} | Risk: {risk_pips} pips | {label}")

                        tg_msg = (
                            f"🟢LTF BUY — {symbol}\n"
                            f"Entry: {entry} | SL: {sl} | Risk: {risk_pips} pips\n"
                            f"Lot: {LOT_SIZE} | Pip: {pip_size}\n"
                            f"RSI: {rsi} | Trend: {trend} | {label}\n"
                            f"Session: {sess_str} | {ts}\n"
                            f"📊 Context: {context}\n"
                            f"TP1: RSI overbought alert | TP2: Opposite signal"
                        )
                        print(tg_msg)
                        print(f"[{symbol}] Sending BUY signal to Telegram…")
                        await send_telegram(tg_msg)
                        print(f"[{symbol}] Telegram BUY signal sent")

                        if is_high_quality(trend_aligned):
                            send_email(f"⭐ HIGH QUALITY BUY — {symbol}", tg_msg)

                        sig_rec = {
                            "symbol": symbol, "type": "BUY", "time": ts,
                            "entry": entry, "sl": sl,
                            "trend_aligned": trend_aligned, "label": label,
                            "session": sess_str, "rsi": rsi, "trend": trend,
                            "context": context,
                        }
                        recent_signals.append(sig_rec)

                        open_trade_record(symbol, "BUY", entry, sl, trend_aligned, label, sess_str)
                        active_trade[symbol] = {
                            "type": "BUY", "entry": entry,
                            "sl": sl, "pip_size": pip_size,
                            "trend_aligned": trend_aligned, "label": label,
                            "session": sess_str,
                            "rsi_alerted": False,
                        }
                        last_signal_time[symbol] = now
                        last_div_time.setdefault(symbol, {})["BULL"] = bull_candle_dt

                # ── SELL ──
                if bear and bear_idx is not None:
                    # Only signal if this is a NEW divergence candle
                    bear_candle_dt = str(df["datetime"].iloc[bear_idx]) if "datetime" in df.columns else str(bear_idx)
                    sym_div = last_div_time.setdefault(symbol, {})
                    if sym_div.get("BEAR") == bear_candle_dt:
                        bear = False  # same candle, skip
                        print(f"[{symbol}] Bearish divergence already signalled for candle {bear_candle_dt}, skipping")
                    else:
                        await check_tp(symbol, "SELL", price)

                if bear and bear_idx is not None:
                    ds = double_confirm(symbol, "SELL")
                    print(f"[{symbol}] SELL double-confirm stack: {signal_stack.get(symbol, [])} → result: {ds}")

                    if ds == "SELL":
                        entry         = price
                        sl            = get_sl_sell(df, symbol)
                        pip_size      = PIP_SIZES.get(symbol, 0.0001)
                        trend_aligned = (trend == "BEARISH")
                        label         = "Trend Aligned Signal" if trend_aligned else "Counter-Trend Signal"
                        ts            = now.strftime("%Y-%m-%d %H:%M UTC")
                        context       = get_market_context(symbol, price, rsi, sma200_val, atr_val, trend)
                        risk_pips     = round((sl - entry) / pip_size, 1)

                        print(f"[{symbol}] Opening SELL trade | Entry: {entry} | SL: {sl} | Risk: {risk_pips} pips | {label}")

                        tg_msg = (
                            f"🔴LTF_ SELL — {symbol}\n"
                            f"Entry: {entry} | SL: {sl} | Risk: {risk_pips} pips\n"
                            f"Lot: {LOT_SIZE} | Pip: {pip_size}\n"
                            f"RSI: {rsi} | Trend: {trend} | {label}\n"
                            f"Session: {sess_str} | {ts}\n"
                            f"📊 Context: {context}\n"
                            f"TP1: RSI oversold alert | TP2: Opposite signal"
                        )
                        print(tg_msg)
                        print(f"[{symbol}] Sending SELL signal to Telegram…")
                        await send_telegram(tg_msg)
                        print(f"[{symbol}] Telegram SELL signal sent")

                        if is_high_quality(trend_aligned):
                            send_email(f"⭐ HIGH QUALITY SELL — {symbol}", tg_msg)

                        sig_rec = {
                            "symbol": symbol, "type": "SELL", "time": ts,
                            "entry": entry, "sl": sl,
                            "trend_aligned": trend_aligned, "label": label,
                            "session": sess_str, "rsi": rsi, "trend": trend,
                            "context": context,
                        }
                        recent_signals.append(sig_rec)

                        open_trade_record(symbol, "SELL", entry, sl, trend_aligned, label, sess_str)
                        active_trade[symbol] = {
                            "type": "SELL", "entry": entry,
                            "sl": sl, "pip_size": pip_size,
                            "trend_aligned": trend_aligned, "label": label,
                            "session": sess_str,
                            "rsi_alerted": False,
                        }
                        last_signal_time[symbol] = now
                        last_div_time.setdefault(symbol, {})["BEAR"] = bear_candle_dt

            save_state(sess_on, sessions)
            await asyncio.sleep(300)

        except Exception as e:
            print(f"Runtime error: {e}")
            await asyncio.sleep(150)


if __name__ == "__main__":
    asyncio.run(main())
