#!/usr/bin/env python3
import sys
import os
import traceback
import signal
import logging
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
        # Setup logging to file and console
        log_filename = f"trading_{now_ist().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s',
            handlers=[
                logging.FileHandler(log_filename, mode='a', encoding='utf-8'),
                logging.StreamHandler(sys.stdout),
            ]
        )

        print(f"✅ Bot started at {now_ist().strftime('%Y-%m-%d %H:%M:%S IST')}")
        logging.info("Bot startup")

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
        
        # Graceful shutdown: convert SIGTERM/SIGINT to KeyboardInterrupt
        def _graceful_stop(signum, frame):
            logging.warning(f"Received signal {signum}, initiating graceful shutdown...")
            raise KeyboardInterrupt()
        try:
            signal.signal(signal.SIGTERM, _graceful_stop)
            signal.signal(signal.SIGINT, _graceful_stop)
        except Exception:
            # Signals might not be available in some environments; ignore
            pass
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
        logging.info("Bot completed successfully")

        # Persist artifacts for CI
        try:
            # Always generate Excel report (even if there are zero signals)
            trader.generate_excel_report()
        except Exception as e:
            logging.error(f"Failed to generate Excel report: {e}")

        try:
            # Write summary for GitHub summary step
            with open("summary.txt", "w", encoding="utf-8") as f:
                f.write("Intraday Nifty Options Trading - Summary\n")
                f.write(f"Finished: {now_ist().strftime('%Y-%m-%d %H:%M:%S IST')}\n")
                f.write(f"Signals: {len(getattr(trader, 'all_signals', []) or [])}\n")
                f.write(f"Trades: {len(getattr(trader, 'trades_today', []) or [])}\n")
                f.write(f"Daily P&L: {getattr(trader, 'daily_pnl', 0.0):+,.0f}\n")
        except Exception as e:
            logging.error(f"Failed to write summary.txt: {e}")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ ERROR: {error_msg}")
        try:
            with open("summary.txt", "w", encoding="utf-8") as f:
                f.write("Intraday Nifty Options Trading - Summary\n")
                f.write(f"Status: ERROR\n")
                f.write(f"Time: {now_ist().strftime('%Y-%m-%d %H:%M:%S IST')}\n")
                f.write(error_msg)
        except Exception:
            pass
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
