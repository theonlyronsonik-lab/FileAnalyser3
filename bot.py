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
import pytz  # Added for Nairobi timezone

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

NAIROBI_TZ = pytz.timezone("Africa/Nairobi")

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
    "Asia":     (0, 10),
    "London":   (10, 15),
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
        "last_scan":        datetime.now(NAIROBI_TZ).strftime("%Y-%m-%d %H:%M Nairobi"),
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
    hour = datetime.now(NAIROBI_TZ).hour
    active = [name for name, (s, e) in SESSIONS.items() if s <= hour <= e]
    return active if active else ["Off-Hours"]

def session_active():
    return get_active_sessions() != ["Off-Hours"]

def session_label(sessions):
    return " / ".join(sessions) if sessions else "Off-Hours"


# ─────────────────────────────────────────────
# MARKET CONTEXT
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

    hour = datetime.now(NAIROBI_TZ).hour
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
    hour = datetime.now(NAIROBI_TZ).hour
    return trend_aligned and (14 <= hour <= 20)


# ─────────────────────────────────────────────
# MAIN LOOP AND EVERYTHING ELSE
# ─────────────────────────────────────────────
    # ─────────────────────────────────────────────
# MAIN LOOP (with Nairobi UTC)
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

    # Nairobi timezone offset
    NAIR_OB_OFFSET = 3

    while True:
        try:
            # Get Nairobi local hour
            now_utc = datetime.now(timezone.utc)
            now_nair = now_utc + timedelta(hours=NAIR_OB_OFFSET)
            hour_nair = now_nair.hour

            # Determine active sessions in Nairobi time
            active_sessions = [
                name for name, (start, end) in SESSIONS.items()
                if start <= hour_nair <= end
            ]
            session_on = bool(active_sessions)
            sess_str = " / ".join(active_sessions) if session_on else "Off-Hours"

            if not session_on:
                now_str = now_nair.strftime("%H:%M")
                print(f"[{now_str} Nairobi] Off-hours, sleeping 60s…")
                save_state(False, active_sessions)
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

                # Check for divergences
                bull, bull_idx = bullish_div(df)
                bear, bear_idx = bearish_div(df)

                if symbol in last_signal_time:
                    if now_nair - last_signal_time[symbol] < timedelta(minutes=COOLDOWN_MINUTES):
                        continue

                ts = now_nair.strftime("%Y-%m-%d %H:%M Nairobi")

                # ── BUY ──
                if bull and bull_idx is not None:
                    await check_tp(symbol, "BUY")
                    ds = double_confirm(symbol, "BUY")

                    if ds == "BUY":
                        entry         = price
                        sl            = round(df["low"].iloc[bull_idx], 5)
                        trend_aligned = (trend == "BULLISH")
                        label         = "Trend Aligned Signal" if trend_aligned else "Counter-Trend Signal"
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
                            "trend_aligned": trend_aligned,
                            "label": label,
                            "session": sess_str,
                            "rsi_alerted": False,
                        }
                        last_signal_time[symbol] = now_nair

                # ── SELL ──
                if bear and bear_idx is not None:
                    await check_tp(symbol, "SELL")
                    ds = double_confirm(symbol, "SELL")

                    if ds == "SELL":
                        entry         = price
                        sl            = round(df["high"].iloc[bear_idx], 5)
                        trend_aligned = (trend == "BEARISH")
                        label         = "Trend Aligned Signal" if trend_aligned else "Counter-Trend Signal"
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
                            "trend_aligned": trend_aligned,
                            "label": label,
                            "session": sess_str,
                            "rsi_alerted": False,
                        }
                        last_signal_time[symbol] = now_nair

            save_state(session_on, active_sessions)
            await asyncio.sleep(60)

        except Exception as e:
            print(f"Error in main loop: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
