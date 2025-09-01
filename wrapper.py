#!/usr/bin/env python3
import sys
import os
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo  # built-in since Python 3.9

from notifications import TradingNotifications
from options_trader_ist import IntradayNiftyTrader  # use your patched IST-safe trader

# Define IST timezone once
IST = ZoneInfo("Asia/Kolkata")

def now_ist():
    """Return current IST datetime"""
    return datetime.now(IST)

def main():
    notifier = TradingNotifications()

    try:
        # Log bot start
        print(f"✅ Bot started at {now_ist().strftime('%Y-%m-%d %H:%M:%S IST')}")

        # Send start notification
        notifier.bot_started()

        # Load token
        with open("nifty_intraday_token.txt", "r") as f:
            token = f.read().strip()

        # Initialize and run trader
        trader = IntradayNiftyTrader(token)
        trader.run()

        # Send completion notification
        notifier.bot_completed(
            len(trader.all_signals),
            len(trader.trades_today),
            trader.daily_pnl,
        )

        print(f"✅ Bot finished at {now_ist().strftime('%Y-%m-%d %H:%M:%S IST')}")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ ERROR: {error_msg}")
        notifier.bot_error(error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
