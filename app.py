import json
import os
from flask import Flask, render_template, jsonify
from datetime import datetime, timezone

app = Flask(__name__)
SIGNALS_FILE = "signals.json"


def load_signals():
    if not os.path.exists(SIGNALS_FILE):
        return {
            "bot_status": "starting",
            "last_scan": None,
            "session_active": False,
            "current_sessions": [],
            "symbols": {},
            "recent_signals": [],
            "trades_history": [],
            "stats": {
                "total": 0, "wins": 0, "losses": 0,
                "pending": 0, "win_rate": 0,
                "by_asset": {}, "by_session": {}
            },
        }
    try:
        with open(SIGNALS_FILE) as f:
            return json.load(f)
    except Exception:
        return {
            "bot_status": "error",
            "symbols": {},
            "recent_signals": [],
            "trades_history": [],
            "stats": {}
        }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data")
def api_data():
    data = load_signals()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    signals_today = sum(
        1 for s in data.get("recent_signals", [])
        if s.get("time", "").startswith(today)
    )
    data["signals_today"] = signals_today
    return jsonify(data)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
