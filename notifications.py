import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

class TradingNotifications:
    def __init__(self):
        self.sender_email = "akumaran313@gmail.com"
        self.sender_password = "nfxv pygy qbbm pcdn"
        self.recipient_email = "akumaran313@gmail.com"
        
    def send_notification(self, subject, body):
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
            return True
        except Exception as e:
            print(f"Failed to send notification: {e}")
            return False
    
    def bot_started(self):
        self.send_notification(
            "Trading Bot Started",
            f"Bot started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nMonitoring markets..."
        )
    
    def bot_completed(self, signals=0, trades=0, pnl=0):
        self.send_notification(
            "Trading Session Completed",
            f"""Session completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary:
- Signals: {signals}
- Trades: {trades}
- P&L: Rs.{pnl:,.2f}

Check GitHub Actions for reports."""
        )
    
    def bot_error(self, error):
        self.send_notification(
            "⚠️ Bot Error",
            f"Error at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{error}"
        )