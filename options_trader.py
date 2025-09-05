import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import json
from tabulate import tabulate
import warnings
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import webbrowser
from urllib.parse import urlparse, parse_qs
import smtplib
import os
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
import sys
import pytz
import itertools
from discord_mail_alerts import NotificationManager
from dotenv import load_dotenv



# Load environment variables from .env if present
load_dotenv()

# Always use IST timezone for consistency on GitHub Actions
IST = pytz.timezone("Asia/Kolkata")

def now_ist():
    return datetime.now(IST)
warnings.filterwarnings('ignore')

class StatusDisplay:
    """Enhanced terminal status display with animations"""
    
    def __init__(self):
        self.spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.market_icons = itertools.cycle(['📈', '📊', '📉', '📊'])
        self.is_running = False
        
    def clear_line(self):
        """Clear current line in terminal"""
        sys.stdout.write('\r' + ' ' * 120 + '\r')
        sys.stdout.flush()
        
    def get_market_status_bar(self, current_time):
        """Generate market status progress bar"""
        market_open = datetime.strptime("09:15", '%H:%M')
        market_close = datetime.strptime("15:30", '%H:%M')
        current = datetime.strptime(current_time, '%H:%M')
        
        # Add today's date
        now = now_ist()
        market_open = market_open.replace(year=now.year, month=now.month, day=now.day, tzinfo=IST)
        market_close = market_close.replace(year=now.year, month=now.month, day=now.day, tzinfo=IST)
        current = current.replace(year=now.year, month=now.month, day=now.day, tzinfo=IST)
        
        if current < market_open:
            return "[⏳ Pre-Market]"
        elif current > market_close:
            return "[🌙 After Hours]"
        else:
            total_minutes = (market_close - market_open).total_seconds() / 60
            elapsed_minutes = (current - market_open).total_seconds() / 60
            progress = int((elapsed_minutes / total_minutes) * 20)
            bar = "█" * progress + "░" * (20 - progress)
            return f"[{bar}]"
    
    def display_market_closed(self, current_time, market_open, market_close):
        """Display market closed status"""
        self.clear_line()
        icon = next(self.spinner)
        
        if current_time < market_open:
            status = f"{icon} 🌅 Pre-Market | Opens in: {market_open} | Current: {current_time}"
        else:
            status = f"{icon} 🌙 Market Closed | Hours: {market_open}-{market_close} | Current: {current_time}"
        
        sys.stdout.write(f"\r{status}")
        sys.stdout.flush()
    
    def display_weekend(self, day_name, date):
        """Display weekend status"""
        self.clear_line()
        icon = next(self.spinner)
        status = f"{icon} 🏖️  Weekend - Markets Closed | {day_name}, {date}"
        sys.stdout.write(f"\r{status}")
        sys.stdout.flush()
    
    def display_live_market(self, current_time, spot_price, ce_premium, pe_premium, 
                           atm_strike, signal_attempts, trades_today, daily_pnl,
                           has_signal=False):
        """Display live market data with enhanced formatting"""
        self.clear_line()
        
        spinner = next(self.spinner)
        market_icon = next(self.market_icons)
        market_bar = self.get_market_status_bar(current_time)
        
        # Build status line
        status_parts = [
            f"{spinner} {market_icon}",
            f"{current_time}",
            market_bar,
            f"NIFTY: ₹{spot_price:.2f}",
            f"ATM {atm_strike}",
            f"CE: ₹{ce_premium:.2f}",
            f"PE: ₹{pe_premium:.2f}"
        ]
        
        # Add attempts if any
        if signal_attempts > 0:
            status_parts.append(f"📊 Attempts: {signal_attempts}")
        
        # Add trades/PnL if any
        if trades_today > 0 or daily_pnl != 0:
            status_parts.append(f"💼 Trades: {trades_today}")
            pnl_icon = "🟢" if daily_pnl >= 0 else "🔴"
            status_parts.append(f"{pnl_icon} P&L: ₹{daily_pnl:+,.0f}")
        
        # Add signal status
        if has_signal:
            status_parts.append("🎯 SIGNAL!")
        else:
            status_parts.append("👀 Scanning...")
        
        status = " | ".join(status_parts)
        sys.stdout.write(f"\r{status}")
        sys.stdout.flush()
    
    def display_position_monitor(self, position_type, entry_price, current_price, 
                                pnl_percent, time_str):
        """Display position monitoring status"""
        self.clear_line()
        
        spinner = next(self.spinner)
        pnl_icon = "🟢" if pnl_percent >= 0 else "🔴"
        
        # Add visual indicator for P&L ranges
        if pnl_percent >= 50:
            strength = "🔥🔥🔥"
        elif pnl_percent >= 30:
            strength = "🔥🔥"
        elif pnl_percent >= 10:
            strength = "🔥"
        elif pnl_percent >= 0:
            strength = "📈"
        elif pnl_percent >= -10:
            strength = "📉"
        elif pnl_percent >= -20:
            strength = "⚠️"
        else:
            strength = "🚨"
        
        status = (f"{spinner} 📊 POSITION: {position_type} | "
                 f"Entry: ₹{entry_price:.2f} | "
                 f"Current: ₹{current_price:.2f} | "
                 f"{pnl_icon} P&L: {pnl_percent:+.1f}% {strength} | "
                 f"⏰ {time_str}")
        
        sys.stdout.write(f"\r{status}")
        sys.stdout.flush()
    
    def display_startup_banner(self):
        """Display startup banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║     📈 INTRADAY NIFTY OPTIONS TRADING SYSTEM 📈               ║
║     Daily Signal Mode - At Least One Signal Guaranteed        ║
║     Performance Optimized Version                             ║
╚═══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def display_signal_alert(self):
        """Display signal alert animation"""
        print("\n")
        print("🚨 " + "="*60 + " 🚨")
        print("📢 SIGNAL DETECTED! CHECK DETAILS BELOW 📢")
        print("🚨 " + "="*60 + " 🚨")
        print("\n")

class IntradayNiftyTrader:
    def __init__(self, access_token):
        self.access_token = access_token
        self.base_url = "https://api.upstox.com"
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        
        # Email configuration (from environment)
        self.sender_email = os.getenv("EMAIL_SENDER", "")
        self.sender_password = os.getenv("EMAIL_PASSWORD", "")
        self.recipient_email = os.getenv("EMAIL_RECIPIENT", "")
        if not self.sender_email or not self.sender_password or not self.recipient_email:
            print("[WARN] Email env vars missing: EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENT")
        
        # Notification Configuration from environment
        self.discord_config = {
            'webhook_url': os.getenv("DISCORD_WEBHOOK_URL", "")
        }
        if not self.discord_config['webhook_url']:
            print("[WARN] Discord env var missing: DISCORD_WEBHOOK_URL")

        self.telegram_config = {
            'bot_token': os.getenv("TELEGRAM_BOT_TOKEN", ""),
            'chat_id': os.getenv("TELEGRAM_CHAT_ID", "")
        }
        if not self.telegram_config['bot_token'] or not self.telegram_config['chat_id']:
            print("[WARN] Telegram env vars missing: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
        
        # Initialize notification manager (Discord + Telegram + Email)
        self.notification_manager = NotificationManager(
            discord_config=self.discord_config,
            telegram_config=self.telegram_config,
            email_config={
                'sender_email': self.sender_email,
                'sender_password': self.sender_password,
                'recipient_email': self.recipient_email,
            }
        )
        
        # Nifty configuration
        self.nifty_symbol = "NSE_INDEX|Nifty 50"
        self.nifty_instrument_key = "NSE_INDEX|Nifty 50"
        self.nifty_lot_size = 75
        
        # Intraday specific timings
        self.market_open = "09:15"
        self.trading_start = "09:30"
        self.last_entry = "15:00"
        self.square_off_start = "15:15"
        self.market_close = "15:30"
        
        # Intraday indicators
        self.rsi_period = 9
        self.ema_fast = 5
        self.ema_slow = 13
        self.atr_period = 10
        self.volume_lookback = 20
        
        # Advanced indicators
        self.bb_period = 20
        self.bb_std = 2
        self.stoch_k = 14
        self.stoch_d = 3
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # Risk management
        self.fixed_lots = 1
        self.max_trades_per_day = 3
        self.daily_loss_limit = 5000
        self.max_premium = 200
        self.min_premium = 30
        self.sl_percent = 25
        self.target1_percent = 50
        self.target2_percent = 100
        
        # Trade tracking
        self.trades_today = []
        self.all_signals = []
        self.potential_signals = []
        self.daily_pnl = 0
        self.last_signal_time = None
        self.signal_cooldown_minutes = 10
        
        # Detailed trade log (per realized exit)
        self.trades_log = []
        
        # Position tracking
        self.open_position = None
        self.position_entry_time = None
        
        # Signal tracking for daily guarantee
        self.best_signal_today = None
        self.signal_attempts = 0
        
        # Status display
        self.status = StatusDisplay()
        
    
    # --- existing methods remain unchanged ---
    
    def check_market_conditions(self):
        """Check whether market conditions allow searching for a new entry.
        Returns (suitable: bool, message: str)
        """
        now = now_ist()
        time_str = now.strftime('%H:%M')
        
        # Respect entry window
        if time_str < self.trading_start:
            return False, f"Waiting for trading start {self.trading_start}"
        if time_str > self.last_entry:
            return False, f"Past last entry time {self.last_entry}"
        if time_str >= self.square_off_start:
            return False, f"Square-off window started {self.square_off_start}"
        
        # Daily risk limits
        if self.max_trades_per_day and len(self.trades_today) >= self.max_trades_per_day:
            return False, "Max trades for the day reached"
        if self.daily_loss_limit and self.daily_pnl <= -self.daily_loss_limit:
            return False, "Daily loss limit reached"
        
        # Cooldown between signals/entries
        if self.last_signal_time is not None:
            elapsed = (now - self.last_signal_time).total_seconds() / 60.0
            remaining = self.signal_cooldown_minutes - elapsed
            if remaining > 0:
                return False, f"Cooldown {remaining:.1f} min remaining"
        
        return True, "OK"
    
    def get_access_token(self):
        """Get new access token"""
        API_KEY = os.getenv("UPSTOX_API_KEY", "")
        API_SECRET = os.getenv("UPSTOX_API_SECRET", "")
        REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI", "http://localhost:8080/callback")
        if not API_KEY or not API_SECRET:
            print("[WARN] Upstox env vars missing: UPSTOX_API_KEY, UPSTOX_API_SECRET")
        
        print("\nGetting Upstox Access Token...")
        
        auth_url = f"https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={API_KEY}&redirect_uri={REDIRECT_URI}"
        print(f"\nOpening browser for login...")
        webbrowser.open(auth_url)
        
        redirected_url = input("\nPaste complete redirected URL: ").strip()
        
        try:
            parsed_url = urlparse(redirected_url)
            auth_code = parse_qs(parsed_url.query)['code'][0]
            
            token_url = "https://api.upstox.com/v2/login/authorization/token"
            payload = {
                'code': auth_code,
                'client_id': API_KEY,
                'client_secret': API_SECRET,
                'redirect_uri': REDIRECT_URI,
                'grant_type': 'authorization_code'
            }
            
            response = requests.post(token_url, data=payload, headers={
                'accept': 'application/json',
                'Content-Type': 'application/x-www-form-urlencoded',
            })
            
            if response.status_code == 200:
                token = response.json().get('access_token')
                with open('nifty_intraday_token.txt', 'w') as f:
                    f.write(token)
                print("\nToken saved!")
                return token
            else:
                print(f"\nError: {response.text}")
                return None
        except Exception as e:
            print(f"\nError: {e}")
            return None
    
    def load_token(self):
        """Load saved token"""
        try:
            with open('nifty_intraday_token.txt', 'r') as f:
                return f.read().strip()
        except:
            return None
    
    def get_current_expiry(self):
        """Get current weekly expiry (Tuesday)"""
        today = now_ist()
        
        if today.weekday() == 5 or today.weekday() == 6:
            days_ahead = (1 - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
        else:
            days_ahead = 1 - today.weekday()
            
            if days_ahead <= 0:
                days_ahead += 7
            elif days_ahead == 0 and today.hour >= 15 and today.minute >= 30:
                days_ahead = 7
        
        expiry = today + timedelta(days=days_ahead)
        return expiry.strftime('%Y-%m-%d'), expiry.strftime('%d%b%y').upper()
    
    def get_nifty_spot_price(self):
        """Get current Nifty spot price"""
        symbols = [
            "NSE_INDEX|Nifty 50",
            "NSE_INDEX|Nifty%2050",
            "NSE_INDEX:Nifty%2050",
            "NSE_INDEX:NIFTY50"
        ]
        
        url = f"{self.base_url}/v2/market-quote/quotes"
        
        for symbol in symbols:
            try:
                params = {'symbol': symbol}
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success' and 'data' in data:
                        market_data = data['data']
                        if symbol in market_data:
                            nifty_data = market_data[symbol]
                            if 'last_price' in nifty_data:
                                return nifty_data['last_price']
                        elif market_data:
                            first_key = list(market_data.keys())[0]
                            if 'last_price' in market_data[first_key]:
                                return market_data[first_key]['last_price']
            except Exception as e:
                continue
        
        return None
    
    def get_option_data(self, strike, option_type, expiry_date):
        """Get specific option data using option chain API"""
        try:
            url = f"{self.base_url}/v2/option/chain"
            params = {
                'instrument_key': 'NSE_INDEX|Nifty 50',
                'expiry_date': expiry_date
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and data.get('data'):
                    option_chain = data['data']
                    
                    for strike_data in option_chain:
                        if strike_data.get('strike_price') == float(strike):
                            if option_type.upper() == 'CE':
                                option_info = strike_data.get('call_options', {})
                            else:
                                option_info = strike_data.get('put_options', {})
                            
                            if option_info:
                                market_data = option_info.get('market_data', {})
                                option_greeks = option_info.get('option_greeks', {})
                                
                                return {
                                    'symbol': f"NIFTY {strike} {option_type} {expiry_date}",
                                    'strike': strike,
                                    'type': option_type,
                                    'premium': market_data.get('ltp', 0),
                                    'bid': market_data.get('bid_price', 0),
                                    'ask': market_data.get('ask_price', 0),
                                    'volume': market_data.get('volume', 0),
                                    'oi': market_data.get('oi', 0),
                                    'iv': option_greeks.get('iv', 0)
                                }
                    
                    return None
                    
        except Exception as e:
            return None
        
        return None
    
    def get_current_day_intraday_candles(self):
        """Get current day intraday candles using intraday API"""
        try:
            instrument_key = "NSE_INDEX%7CNifty%2050"
            interval = "1minute"
            url = f"{self.base_url}/v2/historical-candle/intraday/{instrument_key}/{interval}"
            
            response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and 'data' in data:
                    candles = data['data'].get('candles', [])
                    if len(candles) > 0:
                        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.sort_values('timestamp')
                        df.set_index('timestamp', inplace=True)
                        
                        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'oi']
                        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                        
                        return df
                    else:
                        return None
            else:
                return None
                
        except Exception as e:
            return None

    def resample_to_5min(self, df):
        """Resample 1-minute data to 5-minute candles"""
        try:
            df_5min = df.resample('5T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'oi': 'last'
            })
            df_5min = df_5min.dropna()
            return df_5min
        except Exception as e:
            print(f"Error resampling data: {e}")
            return df
    
    def calculate_rsi(self, prices, period=None):
        """Calculate RSI"""
        if period is None:
            period = self.rsi_period
            
        if len(prices) < period + 1:
            return np.full_like(prices, 50.0)
            
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return np.full_like(prices, 100.0)
        
        rs = up / down
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            
            if down == 0:
                rsi[i] = 100
            else:
                rs = up / down
                rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    def calculate_rsi_optimized(self, prices):
        """Optimized RSI calculation using numpy"""
        if len(prices) < self.rsi_period + 1:
            return np.array([50.0])
        
        # Use numpy for faster calculation
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.convolve(gains, np.ones(self.rsi_period), 'valid') / self.rsi_period
        avg_loss = np.convolve(losses, np.ones(self.rsi_period), 'valid') / self.rsi_period
        
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # Pad with neutral values
        return np.concatenate([np.full(self.rsi_period, 50), rsi])

    def calculate_bollinger_bands_optimized(self, prices):
        """Optimized Bollinger Bands using numpy"""
        prices_array = np.array(prices)
        
        # Use numpy for faster rolling calculations
        middle = np.convolve(prices_array, np.ones(self.bb_period), 'valid') / self.bb_period
        
        # Calculate rolling std manually for speed
        std = np.array([np.std(prices_array[i:i+self.bb_period]) 
                        for i in range(len(prices_array) - self.bb_period + 1)])
        
        upper = middle + (self.bb_std * std)
        lower = middle - (self.bb_std * std)
        
        # Pad to match original length
        pad_length = len(prices) - len(middle)
        upper = np.concatenate([np.full(pad_length, np.nan), upper])
        lower = np.concatenate([np.full(pad_length, np.nan), lower])
        middle = np.concatenate([np.full(pad_length, np.nan), middle])
        
        return upper, middle, lower

    def calculate_vwap_optimized(self, df):
        """Optimized VWAP calculation"""
        # Use last 30 candles for faster calculation
        recent_df = df.tail(30)
        typical_price = (recent_df['high'] + recent_df['low'] + recent_df['close']) / 3
        cumulative_tp_volume = (typical_price * recent_df['volume']).sum()
        cumulative_volume = recent_df['volume'].sum()
        
        return cumulative_tp_volume / cumulative_volume if cumulative_volume > 0 else df['close'].iloc[-1]
    
    def calculate_vwap(self, df):
        """Calculate VWAP"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap
    
    def calculate_atr(self, df, period=None):
        """Calculate ATR"""
        if period is None:
            period = self.atr_period
            
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_bollinger_bands(self, prices):
        """Calculate Bollinger Bands"""
        prices_series = pd.Series(prices)
        middle_bb = prices_series.rolling(self.bb_period).mean()
        std = prices_series.rolling(self.bb_period).std()
        upper_bb = middle_bb + (std * self.bb_std)
        lower_bb = middle_bb - (std * self.bb_std)
        return upper_bb, middle_bb, lower_bb
    
    def calculate_stochastic(self, df):
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=self.stoch_k).min()
        high_max = df['high'].rolling(window=self.stoch_k).max()
        k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=self.stoch_d).mean()
        return k_percent, d_percent
    
    def calculate_macd(self, prices):
        """Calculate MACD"""
        prices_series = pd.Series(prices)
        ema_fast = prices_series.ewm(span=self.macd_fast).mean()
        ema_slow = prices_series.ewm(span=self.macd_slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_market_profile(self, df):
        """Calculate market profile levels"""
        if len(df) < 20:
            return None
            
        today = now_ist().date()
        # Handle timezone-aware index
        if df.index.tz is not None:
            df_dates = df.index.tz_convert(None).date
        else:
            df_dates = df.index.date
        today_data = df[df_dates == today]
        
        if len(today_data) < 5:
            today_data = df.tail(30)
        
        prices = today_data[['high', 'low', 'close']].values.flatten()
        hist, bins = np.histogram(prices, bins=50)
        
        poc_idx = np.argmax(hist)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        total_volume = hist.sum()
        value_area_volume = total_volume * 0.7
        
        current_volume = hist[poc_idx]
        low_idx = poc_idx
        high_idx = poc_idx
        
        while current_volume < value_area_volume and (low_idx > 0 or high_idx < len(hist) - 1):
            if low_idx > 0 and high_idx < len(hist) - 1:
                if hist[low_idx - 1] > hist[high_idx + 1]:
                    low_idx -= 1
                    current_volume += hist[low_idx]
                else:
                    high_idx += 1
                    current_volume += hist[high_idx]
            elif low_idx > 0:
                low_idx -= 1
                current_volume += hist[low_idx]
            elif high_idx < len(hist) - 1:
                high_idx += 1
                current_volume += hist[high_idx]
        
        val = (bins[low_idx] + bins[low_idx + 1]) / 2
        vah = (bins[high_idx] + bins[high_idx + 1]) / 2
        
        return {
            'poc': poc,
            'val': val,
            'vah': vah
        }
    
    def calculate_order_flow_imbalance(self, df):
        """Calculate order flow imbalance"""
        if len(df) < 10:
            return {'imbalance': 0, 'buying_pressure': 0.5, 'selling_pressure': 0.5}
            
        close_price = df['close'].values
        volume = df['volume'].values
        
        price_changes = np.diff(close_price)
        
        buying_volume = 0
        selling_volume = 0
        
        for i in range(len(price_changes)):
            if price_changes[i] > 0:
                buying_volume += volume[i+1]
            elif price_changes[i] < 0:
                selling_volume += volume[i+1]
            else:
                buying_volume += volume[i+1] / 2
                selling_volume += volume[i+1] / 2
        
        total_volume = buying_volume + selling_volume
        
        if total_volume == 0:
            return {'imbalance': 0, 'buying_pressure': 0.5, 'selling_pressure': 0.5}
        
        buying_pressure = buying_volume / total_volume
        selling_pressure = selling_volume / total_volume
        imbalance = buying_pressure - selling_pressure
        
        return {
            'imbalance': imbalance,
            'buying_pressure': buying_pressure,
            'selling_pressure': selling_pressure
        }
    
    def check_divergence(self, prices, rsi_values):
        """Check for RSI divergence"""
        if len(prices) < 20 or len(rsi_values) < 20:
            return None
            
        price_peaks = []
        price_troughs = []
        rsi_peaks = []
        rsi_troughs = []
        
        for i in range(5, len(prices) - 5):
            if prices[i] > max(prices[i-5:i]) and prices[i] > max(prices[i+1:i+6]):
                price_peaks.append((i, prices[i]))
            if prices[i] < min(prices[i-5:i]) and prices[i] < min(prices[i+1:i+6]):
                price_troughs.append((i, prices[i]))
            if rsi_values[i] > max(rsi_values[i-5:i]) and rsi_values[i] > max(rsi_values[i+1:i+6]):
                rsi_peaks.append((i, rsi_values[i]))
            if rsi_values[i] < min(rsi_values[i-5:i]) and rsi_values[i] < min(rsi_values[i+1:i+6]):
                rsi_troughs.append((i, rsi_values[i]))
        
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            if (price_peaks[-1][1] > price_peaks[-2][1] and 
                rsi_peaks[-1][1] < rsi_peaks[-2][1] and
                abs(price_peaks[-1][0] - rsi_peaks[-1][0]) < 5):
                return 'BEARISH'
        
        if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
            if (price_troughs[-1][1] < price_troughs[-2][1] and 
                rsi_troughs[-1][1] > rsi_troughs[-2][1] and
                abs(price_troughs[-1][0] - rsi_troughs[-1][0]) < 5):
                return 'BULLISH'
        
        return None
    
    def detect_candle_patterns(self, df):
        """Detect candlestick patterns"""
        if len(df) < 3:
            return []
            
        patterns = []
        
        last_candles = df.tail(3)
        
        curr_open = last_candles.iloc[-1]['open']
        curr_high = last_candles.iloc[-1]['high']
        curr_low = last_candles.iloc[-1]['low']
        curr_close = last_candles.iloc[-1]['close']
        curr_body = abs(curr_close - curr_open)
        curr_range = curr_high - curr_low
        
        prev_open = last_candles.iloc[-2]['open']
        prev_close = last_candles.iloc[-2]['close']
        prev_body = abs(prev_close - prev_open)
        
        if (curr_body < curr_range * 0.3 and 
            curr_low + curr_body < curr_open and 
            (curr_high - max(curr_open, curr_close)) < curr_body * 0.1):
            patterns.append('HAMMER')
        
        if (curr_body < curr_range * 0.3 and 
            curr_high - curr_body > curr_close and 
            (min(curr_open, curr_close) - curr_low) < curr_body * 0.1):
            patterns.append('SHOOTING_STAR')
        
        if (prev_close < prev_open and 
            curr_close > curr_open and 
            curr_open <= prev_close and 
            curr_close >= prev_open):
            patterns.append('BULLISH_ENGULFING')
        
        if (prev_close > prev_open and 
            curr_close < curr_open and 
            curr_open >= prev_close and 
            curr_close <= prev_open):
            patterns.append('BEARISH_ENGULFING')
        
        return patterns
    
    def calculate_cumulative_delta(self, df):
        """Calculate cumulative delta"""
        if len(df) < 2:
            return pd.Series([0] * len(df), index=df.index)
            
        delta = []
        for i in range(len(df)):
            if i == 0:
                delta.append(0)
            else:
                price_change = df['close'].iloc[i] - df['close'].iloc[i-1]
                if price_change > 0:
                    delta.append(df['volume'].iloc[i])
                elif price_change < 0:
                    delta.append(-df['volume'].iloc[i])
                else:
                    delta.append(0)
        
        cum_delta = pd.Series(delta, index=df.index).cumsum()
        return cum_delta
    
    def get_opening_range(self, df):
        """Get opening range high and low"""
        today = now_ist().date()
        # Convert index to date for comparison if it has timezone info
        if df.index.tz is not None:
            df_dates = df.index.tz_convert(None).date
        else:
            df_dates = df.index.date
        morning_data = df[df_dates == today]
        
        if len(morning_data) > 0:
            opening_range = morning_data.between_time('09:15', '09:30')
            if len(opening_range) > 0:
                return {
                    'high': opening_range['high'].max(),
                    'low': opening_range['low'].min(),
                    'range': opening_range['high'].max() - opening_range['low'].min()
                }
        return None
    
    def get_option_chain(self, expiry_date):
        """Get option chain data"""
        url = f"{self.base_url}/v2/option/chain"
        
        instrument_keys = [
            "NSE_INDEX|Nifty 50",
            "Nifty 50",
            "NIFTY"
        ]
        
        for inst_key in instrument_keys:
            params = {
                'instrument_key': inst_key,
                'expiry_date': expiry_date
            }
            
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success' and 'data' in data:
                        return data.get('data', {})
                else:
                    pass
            except Exception as e:
                continue
        
        return None
    
    def analyze_option_chain_sentiment(self, expiry_date):
        """Analyze option chain for market sentiment"""
        try:
            chain_data = self.get_option_chain(expiry_date)
            if not chain_data:
                return None
            
            spot_price = self.get_nifty_spot_price()
            if not spot_price:
                return None
                
            atm_strike = round(spot_price / 50) * 50
            
            call_oi_total = 0
            put_oi_total = 0
            call_volume_total = 0
            put_volume_total = 0
            max_call_oi_strike = 0
            max_put_oi_strike = 0
            max_call_oi = 0
            max_put_oi = 0
            
            for strike_data in chain_data:
                strike = strike_data.get('strike_price', 0)
                
                if abs(strike - atm_strike) > 250:
                    continue
                
                call_data = strike_data.get('call_options', {})
                if call_data:
                    call_oi = call_data.get('open_interest', 0)
                    call_volume = call_data.get('volume', 0)
                    
                    call_oi_total += call_oi
                    call_volume_total += call_volume
                    
                    if call_oi > max_call_oi:
                        max_call_oi = call_oi
                        max_call_oi_strike = strike
                
                put_data = strike_data.get('put_options', {})
                if put_data:
                    put_oi = put_data.get('open_interest', 0)
                    put_volume = put_data.get('volume', 0)
                    
                    put_oi_total += put_oi
                    put_volume_total += put_volume
                    
                    if put_oi > max_put_oi:
                        max_put_oi = put_oi
                        max_put_oi_strike = strike
            
            pcr_oi = put_oi_total / call_oi_total if call_oi_total > 0 else 0
            pcr_volume = put_volume_total / call_volume_total if call_volume_total > 0 else 0
            
            sentiment = {
                'pcr_oi': pcr_oi,
                'pcr_volume': pcr_volume,
                'max_call_oi_strike': max_call_oi_strike,
                'max_put_oi_strike': max_put_oi_strike,
                'market_bias': 'NEUTRAL'
            }
            
            if pcr_oi > 1.2:
                sentiment['market_bias'] = 'OVERSOLD'
            elif pcr_oi < 0.8:
                sentiment['market_bias'] = 'OVERBOUGHT'
            
            sentiment['immediate_resistance'] = max_call_oi_strike
            sentiment['immediate_support'] = max_put_oi_strike
            
            return sentiment
            
        except Exception as e:
            print(f"Error analyzing option chain: {e}")
            return None
    
    def get_intraday_momentum(self, df):
        """Get intraday momentum indicators"""
        if len(df) < 20:
            return None
            
        current_price = df['close'].iloc[-1]
        
        ema_fast = df['close'].ewm(span=self.ema_fast).mean()
        ema_slow = df['close'].ewm(span=self.ema_slow).mean()
        
        roc = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100
        
        avg_volume = df['volume'].rolling(self.volume_lookback).mean()
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1
        
        vwap = self.calculate_vwap(df)
        
        cum_delta = self.calculate_cumulative_delta(df)
        
        return {
            'ema_fast': ema_fast.iloc[-1],
            'ema_slow': ema_slow.iloc[-1],
            'ema_crossover': ema_fast.iloc[-1] > ema_slow.iloc[-1],
            'roc': roc.iloc[-1],
            'volume_ratio': volume_ratio,
            'vwap': vwap.iloc[-1],
            'price_vs_vwap': (current_price - vwap.iloc[-1]) / vwap.iloc[-1] * 100,
            'cum_delta': cum_delta.iloc[-1],
            'delta_trend': 'BULLISH' if cum_delta.iloc[-1] > cum_delta.iloc[-5] else 'BEARISH'
        }
    
    def select_strike_for_intraday(self, spot_price, option_type, expiry_date):
        """Select best strike for intraday trading - RELAXED criteria"""
        atm_strike = round(spot_price / 50) * 50
        
        if option_type == 'CE':
            strikes = [
                atm_strike,
                atm_strike + 50,
                atm_strike + 100,
                atm_strike - 50,
            ]
        else:
            strikes = [
                atm_strike,
                atm_strike - 50,
                atm_strike - 100,
                atm_strike + 50,
            ]
        
        best_option = None
        candidates = []
        
        for strike in strikes:
            option = self.get_option_data(strike, option_type, expiry_date)
            if option and option['premium'] > 0:
                spread = (option['ask'] - option['bid']) / option['ask'] if option['ask'] > 0 else 1
                moneyness = abs(spot_price - strike) / spot_price * 100
                
                score = 0
                
                if self.min_premium <= option['premium'] <= self.max_premium:
                    score += 3
                elif 20 <= option['premium'] <= 250:
                    score += 2
                elif option['premium'] > 0:
                    score += 1
                
                if option['volume'] > 500:
                    score += 2
                elif option['volume'] > 100:
                    score += 1
                
                if spread < 0.05:
                    score += 2
                elif spread < 0.10:
                    score += 1
                
                if moneyness < 1:
                    score += 2
                elif moneyness < 2:
                    score += 1
                
                candidates.append({
                    'option': option,
                    'score': score,
                    'spread': spread,
                    'moneyness': moneyness
                })
        
        if candidates:
            candidates.sort(key=lambda x: x['score'], reverse=True)
            best_candidate = candidates[0]
            
            if best_candidate['score'] >= 3:
                best_option = best_candidate['option']
            elif best_candidate['score'] >= 2 and len(self.trades_today) == 0:
                best_option = best_candidate['option']
        
        return best_option
    
    def generate_intraday_signal(self, df):
        """Generate intraday trading signals - PERFORMANCE OPTIMIZED"""
        if df is None or len(df) < 30:
            return None
        
        current_time = now_ist().strftime('%H:%M')
        
        # Check trading time window
        if current_time < self.trading_start or current_time > self.last_entry:
            return None
        
        # Check daily limits
        if self.max_trades_per_day and len(self.trades_today) >= self.max_trades_per_day:
            return None
        
        if self.daily_loss_limit and self.daily_pnl <= -self.daily_loss_limit:
            return None
        
        # Check cooldown
        if self.last_signal_time and len(self.trades_today) > 0:
            time_diff = (now_ist() - self.last_signal_time).seconds / 60
            if time_diff < self.signal_cooldown_minutes:
                return None
        
        # ===== PERFORMANCE OPTIMIZATION START =====
        
        # Quick market movement check
        current_price = df['close'].iloc[-1]
        price_5min_ago = df['close'].iloc[-5]
        price_change_5min = ((current_price - price_5min_ago) / price_5min_ago) * 100
        
        # Skip analysis if market is flat (less than 0.05% movement)
        if abs(price_change_5min) < 0.05 and len(self.trades_today) > 0:
            return None
        
        # Calculate only essential indicators first
        close_prices = df['close'].values
        volumes = df['volume'].values
        
        # 1. Volume check (fast)
        avg_volume = np.mean(volumes[-self.volume_lookback:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Early exit if volume too low
        if volume_ratio < 0.5 and len(self.trades_today) > 0:
            return None
        
        # 2. Price momentum (fast)
        roc = price_change_5min
        
        # 3. RSI - Optimized calculation
        rsi = self.calculate_rsi_optimized(close_prices)[-1]
        
        # INCREMENT SIGNAL ATTEMPTS
        self.signal_attempts += 1
        
        # ===== TIERED ANALYSIS APPROACH =====
        
        bull_score = 0
        bear_score = 0
        reasons = []
        confidence_multiplier = 1.0
        
        # Time-based adjustments
        current_hour = int(current_time.split(':')[0])
        if len(self.trades_today) == 0:
            if current_hour >= 14:
                confidence_multiplier *= 1.3
            elif current_hour >= 13:
                confidence_multiplier *= 1.2
        
        # TIER 1: Basic signals (fast calculations only)
        tier1_triggered = False
        
        # RSI signals
        if rsi < 35:
            bull_score += 2.5
            reasons.append(f"RSI oversold ({rsi:.1f})")
            tier1_triggered = True
        elif rsi > 65:
            bear_score += 2.5
            reasons.append(f"RSI overbought ({rsi:.1f})")
            tier1_triggered = True
        
        # Volume spike
        if volume_ratio > 1.5:
            if roc > 0.3:
                bull_score += 2
                reasons.append(f"Volume spike with upward momentum")
                tier1_triggered = True
            elif roc < -0.3:
                bear_score += 2
                reasons.append(f"Volume spike with downward momentum")
                tier1_triggered = True
        
        # If no basic signals and we have trades, skip expensive calculations
        if not tier1_triggered and len(self.trades_today) > 0:
            return None
        
        # TIER 2: EMA calculations (medium cost)
        ema_fast = None
        ema_slow = None
        
        if tier1_triggered or len(self.trades_today) == 0:
            close_series = pd.Series(close_prices)
            ema_fast = close_series.ewm(span=self.ema_fast, adjust=False).mean().iloc[-1]
            ema_slow = close_series.ewm(span=self.ema_slow, adjust=False).mean().iloc[-1]
            
            # EMA signals
            if ema_fast > ema_slow * 1.001:
                if volume_ratio > 1.2 and roc > 0.3:
                    bull_score += 4
                    reasons.append("Strong uptrend with volume")
            elif ema_fast < ema_slow * 0.999:
                if volume_ratio > 1.2 and roc < -0.3:
                    bear_score += 4
                    reasons.append("Strong downtrend with volume")
        
        # Check if we need TIER 3 (expensive calculations)
        current_max_score = max(bull_score, bear_score) * confidence_multiplier
        
        # Calculate minimum score needed
        if '09:30' <= current_time <= '10:00':
            min_score = 4
        elif '10:00' <= current_time <= '11:30':
            min_score = 3.5
        elif '13:00' <= current_time <= '14:00':
            min_score = 3.5
        else:
            min_score = 4
        
        if len(self.trades_today) == 0:
            min_score *= 0.8
        
        # Skip expensive calculations if we already have enough score
        need_more_analysis = current_max_score < min_score * 1.5
        
        # TIER 3: Expensive calculations (only if needed)
        if need_more_analysis or len(self.trades_today) == 0:
            # Opening Range (medium cost)
            orb = self.get_opening_range(df)
            if orb:
                if current_price > orb['high'] and volume_ratio > 1.1:
                    bull_score += 2.5
                    reasons.append(f"ORB Breakout above {orb['high']:.0f}")
                elif current_price < orb['low'] and volume_ratio > 1.1:
                    bear_score += 2.5
                    reasons.append(f"ORB Breakdown below {orb['low']:.0f}")
            
            # Bollinger Bands (only if really needed)
            if current_max_score < min_score:
                upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands_optimized(close_prices)
                
                if current_price <= lower_bb[-1] * 1.001:
                    bull_score += 2
                    reasons.append("At lower Bollinger Band")
                elif current_price >= upper_bb[-1] * 0.999:
                    bear_score += 2
                    reasons.append("At upper Bollinger Band")
            
            # VWAP (calculate only if needed)
            if ema_fast is not None and ema_slow is not None:
                vwap = self.calculate_vwap_optimized(df)
                price_vs_vwap = (current_price - vwap) / vwap * 100
                
                if abs(price_vs_vwap) < 0.2:
                    if roc > 0.2:
                        bull_score += 2
                        reasons.append("Bounce from VWAP")
                    elif roc < -0.2:
                        bear_score += 2
                        reasons.append("Rejection from VWAP")
        
        # Apply confidence multiplier
        bull_score *= confidence_multiplier
        bear_score *= confidence_multiplier
        
        # Track best signal
        if bull_score > 0 or bear_score > 0:
            # Create minimal momentum data
            momentum = {
                'ema_fast': ema_fast if ema_fast else current_price,
                'ema_slow': ema_slow if ema_slow else current_price,
                'ema_crossover': ema_fast > ema_slow if ema_fast and ema_slow else False,
                'roc': roc,
                'volume_ratio': volume_ratio,
                'vwap': current_price,
                'price_vs_vwap': 0,
                'cum_delta': 0,
                'delta_trend': 'NEUTRAL'
            }
            
            potential_signal = {
                'bull_score': bull_score,
                'bear_score': bear_score,
                'reasons': reasons.copy(),
                'spot_price': current_price,
                'momentum': momentum,
                'atr': df['high'].iloc[-10:].max() - df['low'].iloc[-10:].min(),
                'time': now_ist()
            }
            self.potential_signals.append(potential_signal)
            
            if not self.best_signal_today or max(bull_score, bear_score) > max(self.best_signal_today['bull_score'], self.best_signal_today['bear_score']):
                self.best_signal_today = potential_signal
        
        # Generate signal
        signal = None
        if bull_score >= min_score and bull_score > bear_score * 1.1:
            signal = {
                'type': 'CALL',
                'score': bull_score,
                'strength': 'STRONG' if bull_score >= 6 else ('MEDIUM' if bull_score >= 4 else 'WEAK'),
                'spot_price': current_price,
                'reasons': reasons,
                'momentum': self.best_signal_today['momentum'] if self.best_signal_today else {},
                'atr': self.best_signal_today['atr'] if self.best_signal_today else 0,
                'option_sentiment': {'pcr_oi': 1.0, 'market_bias': 'NEUTRAL', 
                                   'immediate_support': current_price * 0.995,
                                   'immediate_resistance': current_price * 1.005},
            }
        elif bear_score >= min_score and bear_score > bull_score * 1.1:
            signal = {
                'type': 'PUT',
                'score': bear_score,
                'strength': 'STRONG' if bear_score >= 6 else ('MEDIUM' if bear_score >= 4 else 'WEAK'),
                'spot_price': current_price,
                'reasons': reasons,
                'momentum': self.best_signal_today['momentum'] if self.best_signal_today else {},
                'atr': self.best_signal_today['atr'] if self.best_signal_today else 0,
                'option_sentiment': {'pcr_oi': 1.0, 'market_bias': 'NEUTRAL',
                                   'immediate_support': current_price * 0.995,
                                   'immediate_resistance': current_price * 1.005},
            }
        
        # Log near misses only if close
        if not signal and max(bull_score, bear_score) >= min_score * 0.8:
            print(f"\n[Near Miss] Bull: {bull_score:.1f}, Bear: {bear_score:.1f}, Required: {min_score:.1f}")
        
        return signal
    
    def calculate_option_targets(self, entry_premium):
        """Calculate option price targets and stop loss"""
        return {
            'stop_loss': entry_premium * (1 - self.sl_percent / 100),
            'target1': entry_premium * (1 + self.target1_percent / 100),
            'target2': entry_premium * (1 + self.target2_percent / 100),
            'trailing_stop': entry_premium * 0.85
        }
    
    def display_signal(self, signal, option_details):
        """Display intraday signal with trade details"""
        self.status.display_signal_alert()
        
        expiry_date, expiry_str = self.get_current_expiry()
        targets = self.calculate_option_targets(option_details['premium'])
        
        quantity = self.fixed_lots * self.nifty_lot_size
        total_investment = option_details['premium'] * quantity
        max_loss = (option_details['premium'] - targets['stop_loss']) * quantity
        target1_profit = (targets['target1'] - option_details['premium']) * quantity
        target2_profit = (targets['target2'] - option_details['premium']) * quantity
        
        print("\n" + "="*80)
        print(f"INTRADAY {signal['type']} SIGNAL - {signal['strength']}")
        print("="*80)
        
        # Signal Information
        signal_table = [
            ["Signal Type", signal['type']],
            ["Signal Strength", f"{signal['strength']} (Score: {signal['score']:.1f})"],
            ["Spot Price", f"Rs.{signal['spot_price']:.2f}"],
            ["Strike Price", f"Rs.{option_details['strike']}"],
            ["Option Premium", f"Rs.{option_details['premium']:.2f}"],
            ["Expiry", expiry_str],
            ["Time", now_ist().strftime('%H:%M:%S')],
            ["Signal Attempts Today", self.signal_attempts]
        ]
        
        print("\nSIGNAL DETAILS:")
        print(tabulate(signal_table, headers=["Parameter", "Value"], tablefmt="grid"))
        
        # Trade Execution Details
        trade_table = [
            ["Lots", f"{self.fixed_lots} lot"],
            ["Quantity", f"{quantity} shares"],
            ["Investment", f"Rs.{total_investment:,.0f}"],
            ["Stop Loss", f"Rs.{targets['stop_loss']:.2f} (-{self.sl_percent}%)"],
            ["Target 1", f"Rs.{targets['target1']:.2f} (+{self.target1_percent}%)"],
            ["Target 2", f"Rs.{targets['target2']:.2f} (+{self.target2_percent}%)"],
            ["Max Loss", f"Rs.{max_loss:,.0f}"],
            ["Target 1 Profit", f"Rs.{target1_profit:,.0f}"],
            ["Target 2 Profit", f"Rs.{target2_profit:,.0f}"]
        ]
        
        print("\nTRADE EXECUTION:")
        print(tabulate(trade_table, headers=["Parameter", "Value"], tablefmt="grid"))
        
        # Momentum Indicators
        mom = signal.get('momentum', {}) or {}
        ema_fast_val = mom.get('ema_fast', signal.get('spot_price', 0.0))
        ema_slow_val = mom.get('ema_slow', signal.get('spot_price', 0.0))
        roc_val = mom.get('roc', 0.0)
        vol_ratio_val = mom.get('volume_ratio', 0.0)
        vwap_val = mom.get('vwap', signal.get('spot_price', 0.0))
        pvv_val = mom.get('price_vs_vwap', 0.0)
        delta_trend_val = mom.get('delta_trend', 'NEUTRAL')

        momentum_table = [
            ["EMA 5", f"Rs.{ema_fast_val:.2f}"],
            ["EMA 13", f"Rs.{ema_slow_val:.2f}"],
            ["Price ROC", f"{roc_val:.2f}%"],
            ["Volume Ratio", f"{vol_ratio_val:.2f}x"],
            ["VWAP", f"Rs.{vwap_val:.2f}"],
            ["Price vs VWAP", f"{pvv_val:.2f}%"],
            ["Delta Trend", delta_trend_val]
        ]
        
        if signal.get('option_sentiment'):
            momentum_table.extend([
                ["PCR (OI)", f"{signal['option_sentiment']['pcr_oi']:.2f}"],
                ["Market Bias", signal['option_sentiment']['market_bias']],
                ["OI Resistance", f"Rs.{signal['option_sentiment']['immediate_resistance']}"],
                ["OI Support", f"Rs.{signal['option_sentiment']['immediate_support']}"]
            ])
        
        print("\nMOMENTUM INDICATORS:")
        print(tabulate(momentum_table, headers=["Indicator", "Value"], tablefmt="grid"))
        
        # Signal Reasons
        print("\nSIGNAL REASONS:")
        for i, reason in enumerate(signal.get('reasons', []) or [], 1):
            print(f"   {i}. {reason}")
        
        # Option Details
        print("\nOPTION LIQUIDITY:")
        liquidity_table = [
            ["Bid", f"Rs.{option_details['bid']:.2f}"],
            ["Ask", f"Rs.{option_details['ask']:.2f}"],
            ["Spread", f"Rs.{option_details['ask'] - option_details['bid']:.2f}"],
            ["Volume", f"{option_details['volume']:,}"],
            ["Open Interest", f"{option_details['oi']:,}"]
        ]
        print(tabulate(liquidity_table, headers=["Parameter", "Value"], tablefmt="grid"))
        
        # Trading Rules
        print("\nINTRADAY RULES:")
        print("1. Exit at stop loss or target - whichever comes first")
        print("2. Trail stop loss to entry after 30% profit")
        print("3. Book 50% at Target 1, rest at Target 2")
        print("4. Square off all positions by 3:15 PM")
        print("5. Maximum 3 trades per day")
        
        print("\n" + "="*80)
    
    def send_all_alerts(self, signal, option_details):
        """Send alerts via Email and Discord"""
        expiry_date, expiry_str = self.get_current_expiry()
        targets = self.calculate_option_targets(option_details['premium'])
        quantity = self.fixed_lots * self.nifty_lot_size
        total_investment = option_details['premium'] * quantity
        
        # Send all notifications through notification manager (includes Email, Discord, Telegram)
        try:
            self.notification_manager.send_all_alerts(
                signal, option_details, targets, quantity, total_investment
            )
        except Exception as e:
            print(f"Notification error: {e}")
    
    def send_email_alert(self, signal, option_details):
        """Send email alert for intraday signal"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            # Safe access with defaults to avoid KeyErrors
            stype = signal.get('type', 'CALL')
            strength = signal.get('strength', 'WEAK')
            score = signal.get('score', 0)
            spot_price = signal.get('spot_price', 0.0)
            momentum = signal.get('momentum', {}) or {}
            ema_crossover = momentum.get('ema_crossover', False)
            volume_ratio = momentum.get('volume_ratio', 0.0)
            vwap_val = momentum.get('vwap', 0.0)
            delta_trend = momentum.get('delta_trend', 'N/A')
            reasons = signal.get('reasons', []) or []
            reasons_text = chr(10).join([f"- {r}" for r in reasons]) if reasons else "No reasons provided."

            # Option details fallbacks
            strike = option_details.get('strike', 0)
            premium = option_details.get('premium', 0.0)

            msg['Subject'] = f"INTRADAY {stype} - {strength} Signal"
            
            expiry_date, expiry_str = self.get_current_expiry()
            targets = self.calculate_option_targets(premium)
            
            quantity = self.fixed_lots * self.nifty_lot_size
            total_investment = premium * quantity
            
            body = f"""
INTRADAY NIFTY OPTIONS SIGNAL
Signal: {stype} - {strength}
Score: {score:.1f}
Time: {now_ist().strftime('%Y-%m-%d %H:%M:%S')}

TRADE DETAILS:
==============
Option Type: {stype}
Strike Price: Rs.{strike}
Premium: Rs.{premium:.2f}
Expiry: {expiry_str}
Quantity: {quantity} shares ({self.fixed_lots} lot)
Investment: Rs.{total_investment:,.0f}

TARGETS & STOP LOSS:
===================
Stop Loss: Rs.{targets['stop_loss']:.2f} (-{self.sl_percent}%)
Target 1: Rs.{targets['target1']:.2f} (+{self.target1_percent}%)
Target 2: Rs.{targets['target2']:.2f} (+{self.target2_percent}%)

MOMENTUM DATA:
=============
Spot Price: Rs.{spot_price:.2f}
EMA Status: {'Bullish' if ema_crossover else 'Bearish'}
Volume Ratio: {volume_ratio:.2f}x
VWAP: Rs.{vwap_val:.2f}
Delta Trend: {delta_trend}

REASONS:
========
{reasons_text}

RULES:
======
1. Exit at SL or Target
2. Trail SL to entry after 30% profit
3. Book 50% at T1, rest at T2
4. Square off by 3:15 PM

Daily Status: Trade {len(self.trades_today) + 1} of {self.max_trades_per_day}
Current P&L: Rs.{self.daily_pnl:,.0f}
Signal Attempts: {self.signal_attempts}

Note: This is an automated intraday signal.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            print("📧 Email alert sent!")
        except Exception as e:
            print(f"Email error: {e}")
    
    def manage_position(self, df):
        """Manage open position - check for exit conditions"""
        if not self.open_position:
            return None
        
        current_time = now_ist()
        time_str = current_time.strftime('%H:%M')
        
        # Get current option price
        current_option = self.get_option_data(
            self.open_position['strike'],
            self.open_position['type'],
            self.open_position['expiry']
        )
        
        if not current_option:
            return None
        
        current_price = current_option['premium']
        entry_price = self.open_position['entry_price']
        
        # Calculate P&L
        pnl_percent = ((current_price - entry_price) / entry_price) * 100
        pnl_amount = (current_price - entry_price) * self.open_position['quantity']
        
        exit_reason = None
        exit_action = None
        
        # Check stop loss
        if current_price <= self.open_position['stop_loss']:
            exit_reason = "Stop Loss Hit"
            exit_action = "FULL_EXIT"
        
        # Check targets
        elif current_price >= self.open_position['target2']:
            exit_reason = "Target 2 Achieved"
            exit_action = "FULL_EXIT"
        
        elif current_price >= self.open_position['target1'] and not self.open_position.get('partial_exit'):
            exit_reason = "Target 1 Achieved"
            exit_action = "PARTIAL_EXIT"
        
        # Time-based exit
        elif time_str >= self.square_off_start:
            exit_reason = "End of Day Square Off"
            exit_action = "FULL_EXIT"
        
        # Trailing stop loss (after 30% profit)
        elif pnl_percent >= 30:
            # Trail to entry price as per rules
            trailing_sl = entry_price
            if current_price <= trailing_sl:
                exit_reason = "Trailing Stop Loss"
                exit_action = "FULL_EXIT"
        
        if exit_action:
            print(f"\n{'='*60}")
            print(f"POSITION EXIT SIGNAL - {exit_reason}")
            print(f"{'='*60}")
            print(f"Entry Price: Rs.{entry_price:.2f}")
            print(f"Current Price: Rs.{current_price:.2f}")
            print(f"P&L: Rs.{pnl_amount:,.0f} ({pnl_percent:+.1f}%)")
            print(f"Action: {exit_action}")
            print(f"Time: {current_time.strftime('%H:%M:%S')}")
            print(f"{'='*60}\n")
            
            # Log realized trade(s) and update daily P&L
            try:
                # Common fields for trade log
                entry_time = self.open_position.get('entry_time')
                trade_base = {
                    'entry_time': entry_time.strftime('%Y-%m-%d %H:%M:%S') if hasattr(entry_time, 'strftime') else str(entry_time),
                    'exit_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'type': self.open_position.get('type'),
                    'strike': self.open_position.get('strike'),
                    'expiry': self.open_position.get('expiry'),
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'exit_reason': exit_reason,
                    'pnl_percent': round(pnl_percent, 2),
                }

                if exit_action == "FULL_EXIT":
                    exit_qty = int(self.open_position.get('quantity', 0))
                    realized_pnl = pnl_amount
                    trade_record = {
                        **trade_base,
                        'quantity': exit_qty,
                        'realized_pnl': round(realized_pnl, 2),
                        'partial': bool(self.open_position.get('partial_exit', False)),
                        'action': exit_action,
                    }
                    self.trades_log.append(trade_record)
                elif exit_action == "PARTIAL_EXIT":
                    # For partial, pnl_amount corresponds to full quantity; realized is 50%
                    exit_qty = max(1, int(self.open_position.get('quantity', 0)) // 2)
                    realized_pnl = pnl_amount * 0.5
                    trade_record = {
                        **trade_base,
                        'quantity': exit_qty,
                        'realized_pnl': round(realized_pnl, 2),
                        'partial': True,
                        'action': exit_action,
                    }
                    self.trades_log.append(trade_record)
            except Exception as _:
                # Do not block on logging errors
                pass

            # Update daily P&L
            if exit_action == "FULL_EXIT":
                self.daily_pnl += pnl_amount
                self.open_position = None
            elif exit_action == "PARTIAL_EXIT":
                self.daily_pnl += pnl_amount * 0.5  # Book 50%
                self.open_position['partial_exit'] = True
                self.open_position['quantity'] = self.open_position['quantity'] // 2
            
            return {
                'action': exit_action,
                'reason': exit_reason,
                'pnl': pnl_amount,
                'pnl_percent': pnl_percent
            }
        
        return None
    
    def run(self):
        """Main intraday trading loop - MODIFIED for daily signals with enhanced status"""
        self.status.display_startup_banner()
        
        # Display configuration
        config_table = [
            ["Trading Hours", f"{self.trading_start} to {self.last_entry}"],
            ["Square Off Time", self.square_off_start],
            ["Lot Size", f"{self.nifty_lot_size} shares"],
            ["Fixed Lots", f"{self.fixed_lots} lot"],
            ["Max Premium", f"Rs.{self.max_premium}"],
            ["Min Premium", f"Rs.{self.min_premium}"],
            ["Stop Loss", f"{self.sl_percent}%"],
            ["Target 1", f"{self.target1_percent}%"],
            ["Target 2", f"{self.target2_percent}%"],
            ["Max Trades/Day", f"{self.max_trades_per_day}"],
            ["Daily Loss Limit", f"Rs.{self.daily_loss_limit}"],
        ]
        
        print("\n📋 CONFIGURATION:")
        print(tabulate(config_table, headers=["Parameter", "Value"], tablefmt="grid"))
        
        print("\n✅ System Ready! Starting market monitoring...")
        print("📌 Press Ctrl+C to stop\n")
        
        check_interval = 15  # Check every 15 seconds
        
        while True:
            try:
                now = now_ist()
                current_time = now.strftime('%H:%M')
                
                # Weekend check
                if now.weekday() == 5 or now.weekday() == 6:
                    self.status.display_weekend(
                        now.strftime('%A'), 
                        now.strftime('%Y-%m-%d')
                    )
                    time.sleep(60)
                    continue
                
                # Market hours check
                if current_time < self.market_open or current_time > self.market_close:
                    self.status.display_market_closed(
                        current_time, 
                        self.market_open, 
                        self.market_close
                    )
                    time.sleep(60)
                    continue
                
                # Reset daily counters at market open
                if current_time == self.market_open and len(self.trades_today) > 0:
                    self.trades_today = []
                    self.daily_pnl = 0
                    self.signal_attempts = 0
                    self.best_signal_today = None
                    self.potential_signals = []
                    print("\n🔄 Daily counters reset for new trading day\n")
                
                # Manage existing position
                if self.open_position:
                    df = self.get_current_day_intraday_candles()
                    if df is not None:
                        # Get current option price for display
                        current_option = self.get_option_data(
                            self.open_position['strike'],
                            self.open_position['type'],
                            self.open_position['expiry']
                        )
                        
                        if current_option:
                            current_price = current_option['premium']
                            entry_price = self.open_position['entry_price']
                            pnl_percent = ((current_price - entry_price) / entry_price) * 100
                            
                            self.status.display_position_monitor(
                                self.open_position['type'],
                                entry_price,
                                current_price,
                                pnl_percent,
                                current_time
                            )
                        
                        exit_signal = self.manage_position(df)
                        if exit_signal:
                            self.last_signal_time = now
                
                # Check for new signals
                elif current_time >= self.trading_start:
                    # Check market conditions
                    suitable, condition_msg = self.check_market_conditions()
                    if not suitable:
                        self.status.clear_line()
                        sys.stdout.write(f"\r⚠️  Skipping - {condition_msg}")
                        sys.stdout.flush()
                        time.sleep(check_interval)
                        continue
                    
                    # Get market data
                    spot_price = self.get_nifty_spot_price()
                    if spot_price:
                        atm_strike = round(spot_price / 50) * 50
                        expiry_date, _ = self.get_current_expiry()
                        
                        # Get ATM option premiums
                        ce_data = self.get_option_data(atm_strike, 'CE', expiry_date)
                        pe_data = self.get_option_data(atm_strike, 'PE', expiry_date)
                        
                        ce_premium = ce_data.get('premium', 0) if ce_data else 0
                        pe_premium = pe_data.get('premium', 0) if pe_data else 0
                        
                        # Display live market data
                        self.status.display_live_market(
                            current_time,
                            spot_price,
                            ce_premium,
                            pe_premium,
                            atm_strike,
                            self.signal_attempts,
                            len(self.trades_today),
                            self.daily_pnl,
                            False
                        )
                    
                    # Get data and analyze
                    df = self.get_current_day_intraday_candles()
                    
                    if df is None:
                        time.sleep(check_interval)
                        continue
                    
                    signal = None
                    
                    # Try to get normal signal first
                    if df is not None:
                        signal = self.generate_intraday_signal(df)
                    
                    if signal:
                        # Get option details
                        expiry_date, _ = self.get_current_expiry()
                        option_details = self.select_strike_for_intraday(
                            signal['spot_price'],
                            'CE' if signal['type'] == 'CALL' else 'PE',
                            expiry_date
                        )
                        
                        if option_details:
                            print("\n")  # Clear the status line
                            self.display_signal(signal, option_details)
                            self.send_all_alerts(signal, option_details)
                            
                            # Create position entry
                            targets = self.calculate_option_targets(option_details['premium'])
                            quantity = self.fixed_lots * self.nifty_lot_size
                            
                            self.open_position = {
                                'type': option_details['type'],
                                'strike': option_details['strike'],
                                'expiry': expiry_date,
                                'entry_price': option_details['premium'],
                                'quantity': quantity,
                                'stop_loss': targets['stop_loss'],
                                'target1': targets['target1'],
                                'target2': targets['target2'],
                                'entry_time': now
                            }
                            
                            self.trades_today.append({
                                'time': now,
                                'signal': signal,
                                'option': option_details
                            })
                            
                            # Store signal
                            signal_record = {
                                'time': now,
                                'type': signal['type'],
                                'spot_price': signal['spot_price'],
                                'strike': option_details['strike'],
                                'premium': option_details['premium'],
                                'score': signal['score'],
                                'strength': signal['strength'],
                                'target1': targets['target1'],
                                'target2': targets['target2'],
                                'stop_loss': targets['stop_loss'],
                                'reasons': signal['reasons']
                            }
                            self.all_signals.append(signal_record)
                            
                            self.last_signal_time = now
                            print(f"\n✅ Position opened. Monitoring for exits...")
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                print("\n\n🛑 Shutting down...")
                if self.open_position:
                    print("⚠️  WARNING: Open position exists! Please close manually.")
                print(f"📊 Final Daily P&L: Rs.{self.daily_pnl:+,.0f}")
                print(f"📈 Total Signals Generated: {len(self.all_signals)}")
                print(f"🎯 Total Trades Executed: {len(self.trades_today)}")
                
                # Generate Excel report (always)
                self.generate_excel_report()
                
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                time.sleep(60)


def main():
    """Main entry point"""
    try:
        print("Starting Intraday Nifty Options Trading System...")
        print("Importing libraries...")
        
        # Check for saved token
        trader = IntradayNiftyTrader("")
        
        if os.path.exists('nifty_intraday_token.txt'):
            print("Using saved token...")
            trader.access_token = trader.load_token()
            trader.headers['Authorization'] = f'Bearer {trader.access_token}'
        else:
            print("\nNo saved token found.")
            token = trader.get_access_token()
            if token:
                trader.access_token = token
                trader.headers['Authorization'] = f'Bearer {token}'
            else:
                print("Failed to get access token")
                return
        
        # Run the trader
        trader.run()
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()