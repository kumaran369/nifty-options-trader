import requests
from datetime import datetime

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
🎯 **Target 1:** ₹{t1:.2f} (+50%)
🎯 **Target 2:** ₹{t2:.2f} (+100%)

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

## Removed unused WhatsApp and Telegram classes; Discord-only implementation retained.


class NotificationManager:
    """Manage notification channels (Discord only)"""
    
    def __init__(self, telegram_config=None, discord_config=None, whatsapp_config=None):
        self.discord_webhook = None
        # Initialize Discord webhook
        if discord_config and discord_config.get('webhook_url'):
            self.discord_webhook = DiscordWebhook(discord_config['webhook_url'])
    
    def send_all_alerts(self, signal, option_details, targets, quantity, total_investment):
        """Send alerts to configured channels (Discord only)"""
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
