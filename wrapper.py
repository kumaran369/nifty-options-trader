#!/usr/bin/env python3
import sys
import os
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo  # built-in since Python 3.9
from dotenv import load_dotenv

from options_trader import IntradayNiftyTrader

# Define IST timezone once
IST = ZoneInfo("Asia/Kolkata")

def now_ist():
    """Return current IST datetime"""
    return datetime.now(IST)

def main():
    load_dotenv()

    try:
        print(f"✅ Bot started at {now_ist().strftime('%Y-%m-%d %H:%M:%S IST')}")

        # Load token
        token_path = "nifty_intraday_token.txt"
        if os.path.exists(token_path):
            with open(token_path, "r") as f:
                token = f.read().strip()
        else:
            print("❌ Access token not found. Please generate a new token using IntradayNiftyTrader.get_access_token() from options_trader.py")
            sys.exit(1)

        # Initialize trader
        trader = IntradayNiftyTrader(token)
        # Notify started
        try:
            trader.notification_manager.send_bot_started()
        except Exception:
            pass
        # Run trader
        trader.run()

        # Notify completed
        try:
            trader.notification_manager.send_bot_completed(
                signals=len(getattr(trader, 'all_signals', []) or []),
                trades=len(getattr(trader, 'trades_today', []) or []),
                pnl=getattr(trader, 'daily_pnl', 0.0)
            )
        except Exception:
            pass
        print(f"✅ Bot finished at {now_ist().strftime('%Y-%m-%d %H:%M:%S IST')}")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ ERROR: {error_msg}")
        # Attempt to notify error via email/telegram/discord
        try:
            # Load token if possible to init trader and reuse its notifier
            token = None
            token_path = "nifty_intraday_token.txt"
            if os.path.exists(token_path):
                with open(token_path, "r") as f:
                    token = f.read().strip()
            trader = IntradayNiftyTrader(token or "")
            trader.notification_manager.send_bot_error(error_msg)
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
