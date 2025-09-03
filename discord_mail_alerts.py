import requests
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class DiscordWebhook:
    """Free Discord webhook for trading alerts"""
    
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
    
    def send_message(self, message):
        """Send message to Discord via webhook"""
        try:
            payload = {
                'content': message,
                'username': 'Trading Bot'
            }
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            
            if response.status_code == 204:
                return True, "Discord message sent successfully"
            else:
                return False, f"Error: {response.status_code}"
                
        except Exception as e:
            return False, f"Exception: {str(e)}"
    
    def send_signal_alert(self, signal, option_details, targets, quantity, total_investment):
        """Send formatted trading signal to Discord"""
        try:
            forced_text = "🚨 **FORCED SIGNAL - End of day trade**\n\n" if signal.get('forced') else ""

            s_type = signal.get('type', 'CALL')
            strength = signal.get('strength', 'WEAK')
            score = signal.get('score', 0.0)
            spot = signal.get('spot_price', 0.0)
            momentum = signal.get('momentum', {}) or {}
            volume_ratio = momentum.get('volume_ratio', 0.0)
            vwap_val = momentum.get('vwap', 0.0)

            strike = option_details.get('strike', 0)
            premium = option_details.get('premium', 0.0)
            sl = targets.get('stop_loss', 0.0)
            t1 = targets.get('target1', 0.0)
            t2 = targets.get('target2', 0.0)
            # Dynamic target percentages
            pct1 = ((t1 / premium) - 1.0) * 100 if premium else 0.0
            pct2 = ((t2 / premium) - 1.0) * 100 if premium else 0.0

            message = f"""🚀 **NIFTY OPTIONS SIGNAL** 🚀
{forced_text}📊 **Signal:** {s_type} - {strength}
⭐ **Score:** {score:.1f}
🕐 **Time:** {datetime.now().strftime('%H:%M:%S')}

💰 **TRADE DETAILS:**
━━━━━━━━━━━━━━━━━━━━
📈 **Option:** {s_type}
🎯 **Strike:** ₹{strike}
💵 **Premium:** ₹{premium:.2f}
📦 **Quantity:** {quantity} shares
💸 **Investment:** ₹{total_investment:,.0f}

🎯 **TARGETS & STOP LOSS:**
━━━━━━━━━━━━━━━━━━━━━━━━
🛑 **Stop Loss:** ₹{sl:.2f}
🎯 **Target 1:** ₹{t1:.2f} (+{pct1:.0f}%)
🎯 **Target 2:** ₹{t2:.2f} (+{pct2:.0f}%)

📊 **MARKET DATA:**
━━━━━━━━━━━━━━━━━━━
📍 **Spot:** ₹{spot:.2f}
📈 **Volume:** {volume_ratio:.2f}x
💹 **VWAP:** ₹{vwap_val:.2f}

🔍 **REASONS:**
"""
            
            for reason in signal.get('reasons', []) or []:
                message += f"• {reason}\n"
            
            message += """
⚡ **RULES:**
1️⃣ Exit at SL or Target
2️⃣ Trail SL after 30% profit
3️⃣ Book 50% at T1, rest at T2
4️⃣ Square off by 3:15 PM

🤖 *Automated Signal*"""
            
            success, result = self.send_message(message)
            return success, result
            
        except Exception as e:
            return False, f"Error formatting Discord message: {str(e)}"

class TelegramBot:
    """Telegram bot sender for trading alerts"""
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        # Support multiple chat IDs (comma-separated string or list)
        if isinstance(chat_id, str):
            self.chat_ids = [id.strip() for id in chat_id.split(',') if id.strip()]
        elif isinstance(chat_id, list):
            self.chat_ids = chat_id
        else:
            self.chat_ids = [str(chat_id)]
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    def send_message(self, message):
        """Send plain text message to all configured Telegram chats"""
        all_success = True
        results = []
        
        for chat_id in self.chat_ids:
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            last_err = None
            success = False
            
            for attempt in range(3):
                try:
                    response = requests.post(self.base_url, data=payload, timeout=20)
                    if response.status_code == 200:
                        success = True
                        results.append(f"✅ Chat {chat_id}: Success")
                        break
                    else:
                        last_err = f"Error: {response.status_code} {response.text}"
                except Exception as e:
                    last_err = f"Exception: {str(e)}"
                # small backoff
                try:
                    import time as _t; _t.sleep(1 + attempt)
                except Exception:
                    pass
            
            if not success:
                all_success = False
                results.append(f"❌ Chat {chat_id}: {last_err}")
        
        result_msg = "; ".join(results)
        return all_success, result_msg

    def send_signal_alert(self, signal, option_details, targets, quantity, total_investment):
        """Send formatted trading signal to Telegram"""
        try:
            forced_text = "🚨 <b>FORCED SIGNAL - End of day trade</b>\n\n" if signal.get('forced') else ""

            s_type = signal.get('type', 'CALL')
            strength = signal.get('strength', 'WEAK')
            score = signal.get('score', 0.0)
            spot = signal.get('spot_price', 0.0)
            momentum = signal.get('momentum', {}) or {}
            volume_ratio = momentum.get('volume_ratio', 0.0)
            vwap_val = momentum.get('vwap', 0.0)

            strike = option_details.get('strike', 0)
            premium = option_details.get('premium', 0.0)
            sl = targets.get('stop_loss', 0.0)
            t1 = targets.get('target1', 0.0)
            t2 = targets.get('target2', 0.0)
            # Dynamic target percentages
            pct1 = ((t1 / premium) - 1.0) * 100 if premium else 0.0
            pct2 = ((t2 / premium) - 1.0) * 100 if premium else 0.0

            message = f"""🚀 <b>NIFTY OPTIONS SIGNAL</b> 🚀
{forced_text}📊 <b>Signal:</b> {s_type} - {strength}
⭐ <b>Score:</b> {score:.1f}
🕐 <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}

💰 <b>TRADE DETAILS:</b>
━━━━━━━━━━━━━━━━━━━━
📈 <b>Option:</b> {s_type}
🎯 <b>Strike:</b> ₹{strike}
💵 <b>Premium:</b> ₹{premium:.2f}
📦 <b>Quantity:</b> {quantity} shares
💸 <b>Investment:</b> ₹{total_investment:,.0f}

🎯 <b>TARGETS & STOP LOSS:</b>
━━━━━━━━━━━━━━━━━━━━━━━━
🛑 <b>Stop Loss:</b> ₹{sl:.2f}
🎯 <b>Target 1:</b> ₹{t1:.2f} (+{pct1:.0f}%)
🎯 <b>Target 2:</b> ₹{t2:.2f} (+{pct2:.0f}%)

📊 <b>MARKET DATA:</b>
━━━━━━━━━━━━━━━━━━━
📍 <b>Spot:</b> ₹{spot:.2f}
📈 <b>Volume:</b> {volume_ratio:.2f}x
💹 <b>VWAP:</b> ₹{vwap_val:.2f}

🔍 <b>REASONS:</b>
"""
            for reason in (signal.get('reasons', []) or []):
                message += f"• {reason}\n"

            message += """
⚡ <b>RULES:</b>
1️⃣ Exit at SL or Target
2️⃣ Trail SL after 30% profit
3️⃣ Book 50% at T1, rest at T2
4️⃣ Square off by 3:15 PM

🤖 <i>Automated Signal</i>"""

            return self.send_message(message)
        except Exception as e:
            return False, f"Error formatting Telegram message: {str(e)}"


class NotificationManager:
    """Manage notification channels (Discord + Telegram + Email)"""
    
    def __init__(self, telegram_config=None, discord_config=None, whatsapp_config=None, email_config=None):
        self.discord_webhook = None
        self.telegram_bot = None
        self.email_sender = None
        # Initialize Discord webhook
        if discord_config and discord_config.get('webhook_url'):
            self.discord_webhook = DiscordWebhook(discord_config['webhook_url'])
        # Initialize Telegram bot
        if telegram_config and telegram_config.get('bot_token') and telegram_config.get('chat_id'):
            self.telegram_bot = TelegramBot(
                telegram_config['bot_token'], telegram_config['chat_id']
            )
        # Initialize Email sender
        if email_config and email_config.get('sender_email') and email_config.get('sender_password') and email_config.get('recipient_email'):
            self.email_sender = EmailSender(
                email_config['sender_email'],
                email_config['sender_password'],
                email_config['recipient_email']
            )
    
    def send_all_alerts(self, signal, option_details, targets, quantity, total_investment):
        """Send alerts to configured channels (Discord + Telegram + Email)"""
        results = {}
        # Send Discord alert
        if self.discord_webhook:
            try:
                success, message = self.discord_webhook.send_signal_alert(
                    signal, option_details, targets, quantity, total_investment
                )
                results['discord'] = {'success': success, 'message': message}
                if success:
                    print("💬 Discord alert sent!")
                else:
                    print(f"❌ Discord error: {message}")
            except Exception as e:
                results['discord'] = {'success': False, 'message': str(e)}
                print(f"❌ Discord exception: {e}")
        # Send Telegram alert
        if self.telegram_bot:
            try:
                success, message = self.telegram_bot.send_signal_alert(
                    signal, option_details, targets, quantity, total_investment
                )
                results['telegram'] = {'success': success, 'message': message}
                if success:
                    print("📨 Telegram alert sent!")
                else:
                    print(f"❌ Telegram error: {message}")
            except Exception as e:
                results['telegram'] = {'success': False, 'message': str(e)}
                print(f"❌ Telegram exception: {e}")
        # Send Email alert
        if self.email_sender:
            try:
                subject, body = self._format_email_signal(signal, option_details, targets, quantity, total_investment)
                success, message = self.email_sender.send_email(subject, body)
                results['email'] = {'success': success, 'message': message}
                if success:
                    print("✉️ Email alert sent!")
                else:
                    print(f"❌ Email error: {message}")
            except Exception as e:
                results['email'] = {'success': False, 'message': str(e)}
                print(f"❌ Email exception: {e}")
        return results
    
    def send_position_update(self, position_type, entry_price, current_price, pnl_percent, pnl_amount, action=None):
        """Send position update alerts"""
        pnl_icon = "\ud83d\udfe2" if pnl_percent >= 0 else "\ud83d\udd34"
        action_text = f"\n\ud83d\udea8 ACTION: {action}" if action else ""
        
        message = f"""📊 POSITION UPDATE
━━━━━━━━━━━━━━━━━━━━
📈 Type: {position_type}
💵 Entry: ₹{entry_price:.2f}
💰 Current: ₹{current_price:.2f}
{pnl_icon} P&L: {pnl_percent:+.1f}% (₹{pnl_amount:+,.0f}){action_text}
🕐 Time: {datetime.now().strftime('%H:%M:%S')}"""
        
        # Send to Discord
        if self.discord_webhook:
            self.discord_webhook.send_message(message)
        # Send to Telegram
        if self.telegram_bot:
            self.telegram_bot.send_message(message)
        # Send to Email
        if self.email_sender:
            subject = f"Position Update: {position_type} {pnl_percent:+.1f}%"
            body = message.replace('━━━━━━━━━━━━━━━━━━━━', '-'*20)
            self.email_sender.send_email(subject, body)

    # --- Bot lifecycle notifications ---
    def send_bot_started(self):
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg = f"✅ Trading Bot Started\n🕐 Time: {time_str}"
        # Discord
        if self.discord_webhook:
            self.discord_webhook.send_message(msg)
        # Telegram
        if self.telegram_bot:
            self.telegram_bot.send_message(msg)
        # Email
        if self.email_sender:
            self.email_sender.send_email("Trading Bot Started", msg)

    def send_bot_completed(self, signals=0, trades=0, pnl=0.0):
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg = (
            f"✅ Trading Session Completed\n🕐 Time: {time_str}\n\n"
            f"Summary:\n- Signals: {signals}\n- Trades: {trades}\n- P&L: ₹{pnl:,.2f}"
        )
        if self.discord_webhook:
            self.discord_webhook.send_message(msg)
        if self.telegram_bot:
            self.telegram_bot.send_message(msg)
        if self.email_sender:
            self.email_sender.send_email("Trading Session Completed", msg)

    def send_bot_error(self, error_text):
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg = f"⚠️ Bot Error\n🕐 Time: {time_str}\n\n{error_text}"
        if self.discord_webhook:
            self.discord_webhook.send_message(msg)
        if self.telegram_bot:
            self.telegram_bot.send_message(msg)
        if self.email_sender:
            self.email_sender.send_email("Bot Error", msg)

    def _format_email_signal(self, signal, option_details, targets, quantity, total_investment):
        """Create subject/body for email signal"""
        s_type = signal.get('type', 'CALL')
        strength = signal.get('strength', 'WEAK')
        score = signal.get('score', 0.0)
        spot = signal.get('spot_price', 0.0)
        momentum = signal.get('momentum', {}) or {}
        volume_ratio = momentum.get('volume_ratio', 0.0)
        vwap_val = momentum.get('vwap', 0.0)
        strike = option_details.get('strike', 0)
        premium = option_details.get('premium', 0.0)
        sl = targets.get('stop_loss', 0.0)
        t1 = targets.get('target1', 0.0)
        t2 = targets.get('target2', 0.0)
        # Dynamic target percentages
        pct1 = ((t1 / premium) - 1.0) * 100 if premium else 0.0
        pct2 = ((t2 / premium) - 1.0) * 100 if premium else 0.0
        reasons = '\n'.join([f"- {r}" for r in (signal.get('reasons', []) or [])])
        forced = ' [FORCED]' if signal.get('forced') else ''
        subject = f"NIFTY SIGNAL{forced}: {s_type} {strength} | Strike {strike} | Prem ₹{premium:.2f}"
        body = f"""
NIFTY OPTIONS SIGNAL{forced}
Time: {datetime.now().strftime('%H:%M:%S')}
Signal: {s_type} - {strength}
Score: {score:.1f}

TRADE DETAILS
-------------
Option: {s_type}
Strike: ₹{strike}
Premium: ₹{premium:.2f}
Quantity: {quantity}
Investment: ₹{total_investment:,.0f}

TARGETS & STOP LOSS
-------------------
Stop Loss: ₹{sl:.2f}
Target 1: ₹{t1:.2f} (+{pct1:.0f}%)
Target 2: ₹{t2:.2f} (+{pct2:.0f}%)

MARKET DATA
-----------
Spot: ₹{spot:.2f}
Volume: {volume_ratio:.2f}x
VWAP: ₹{vwap_val:.2f}

REASONS
-------
{reasons}

Rules:
1) Exit at SL or Target
2) Trail SL after 30% profit
3) Book 50% at T1, rest at T2
4) Square off by 3:15 PM
""".strip()
        return subject, body

class EmailSender:
    """Simple Gmail SMTP sender"""
    def __init__(self, sender_email, sender_password, recipient_email):
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email

    def send_email(self, subject, body):
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = f"[Trading Bot] {subject}"
            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            return True, 'Email sent'
        except Exception as e:
            return False, str(e)
