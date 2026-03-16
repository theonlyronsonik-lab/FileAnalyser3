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

SESSIONS = {
    "Asia":     (2,  10),
    "London":   (7,  16),
    "New York": (13, 22),
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
        "last_scan":        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
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
    elif rsi >= 60:
        tips.append(f"RSI {rsi:.1f} — elevated, strong momentum but watch for pullback")
    elif rsi <= RSI_OVERSOLD:
        tips.append(f"RSI {rsi:.1f} — oversold, potential bounce zone")
    elif rsi <= 40:
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
    return trend_aligned and (14 <= hour <= 20)


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


def close_trade_record(symbol, outcome):
    for t in reversed(trades_history):
        if t["symbol"] == symbol and t["outcome"] == "OPEN":
            t["outcome"]    = outcome
            t["close_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            print(f"Trade closed: {symbol} → {outcome}")
            return


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
# TP CHECK — Opposite Signal (TP Model 2)
# ─────────────────────────────────────────────

async def check_tp(symbol, signal):
    if symbol not in active_trade:
        return

    trade = active_trade[symbol]

    if signal == "BUY" and trade["type"] == "SELL":
        msg = (
            f"✅ TP HIT (Opposite Signal) — {symbol}\n"
            f"SELL trade @ {trade['entry']} closed | Outcome: WIN"
        )
        print(msg)
        await send_telegram(msg)
        close_trade_record(symbol, "WIN")
        del active_trade[symbol]

    elif signal == "SELL" and trade["type"] == "BUY":
        msg = (
            f"✅ TP HIT (Opposite Signal) — {symbol}\n"
            f"BUY trade @ {trade['entry']} closed | Outcome: WIN"
        )
        print(msg)
        await send_telegram(msg)
        close_trade_record(symbol, "WIN")
        del active_trade[symbol]


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

                # Check RSI TP zone for active trade (TP Model 1)
                await check_rsi_tp_zone(symbol, rsi)

                bull, bull_idx = bullish_div(df)
                bear, bear_idx = bearish_div(df)

                now = datetime.now(timezone.utc)

                if symbol in last_signal_time:
                    if now - last_signal_time[symbol] < timedelta(minutes=COOLDOWN_MINUTES):
                        continue

                # ── BUY ──
                if bull and bull_idx is not None:
                    await check_tp(symbol, "BUY")
                    ds = double_confirm(symbol, "BUY")

                    if ds == "BUY":
                        entry         = price
                        # SL at the wick (low) of the divergence candle
                        sl            = round(df["low"].iloc[bull_idx], 5)
                        trend_aligned = (trend == "BULLISH")
                        label         = "Trend Aligned Signal" if trend_aligned else "Counter-Trend Signal"
                        ts            = now.strftime("%Y-%m-%d %H:%M UTC")
                        context       = get_market_context(symbol, price, rsi, sma200_val, atr_val, trend)

                        tg_msg = (
                            f"🟢 BUY — {symbol}\n"
                            f"Entry: {entry} | SL: {sl}\n"
                            f"RSI: {rsi} | Trend: {trend} | {label}\n"
                            f"Session: {sess_str} | {ts}\n"
                            f"📊 Context: {context}\n"
                            f"TP1: RSI overbought alert | TP2: Opposite signal"
                        )
                        print(tg_msg)
                        await send_telegram(tg_msg)

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
                            "sl": sl,
                            "trend_aligned": trend_aligned, "label": label,
                            "session": sess_str,
                            "rsi_alerted": False,
                        }
                        last_signal_time[symbol] = now

                # ── SELL ──
                if bear and bear_idx is not None:
                    await check_tp(symbol, "SELL")
                    ds = double_confirm(symbol, "SELL")

                    if ds == "SELL":
                        entry         = price
                        # SL at the wick (high) of the divergence candle
                        sl            = round(df["high"].iloc[bear_idx], 5)
                        trend_aligned = (trend == "BEARISH")
                        label         = "Trend Aligned Signal" if trend_aligned else "Counter-Trend Signal"
                        ts            = now.strftime("%Y-%m-%d %H:%M UTC")
                        context       = get_market_context(symbol, price, rsi, sma200_val, atr_val, trend)

                        tg_msg = (
                            f"🔴 SELL — {symbol}\n"
                            f"Entry: {entry} | SL: {sl}\n"
                            f"RSI: {rsi} | Trend: {trend} | {label}\n"
                            f"Session: {sess_str} | {ts}\n"
                            f"📊 Context: {context}\n"
                            f"TP1: RSI oversold alert | TP2: Opposite signal"
                        )
                        print(tg_msg)
                        await send_telegram(tg_msg)

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
                            "sl": sl,
                            "trend_aligned": trend_aligned, "label": label,
                            "session": sess_str,
                            "rsi_alerted": False,
                        }
                        last_signal_time[symbol] = now

            save_state(sess_on, sessions)
            await asyncio.sleep(300)

        except Exception as e:
            print(f"Runtime error: {e}")
            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
