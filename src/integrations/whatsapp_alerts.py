"""
WhatsApp Alert Integration using Twilio
Sends critical failure predictions via WhatsApp
"""

import os
import logging
from typing import Optional
from twilio.rest import Client
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhatsAppAlerter:
    """Send WhatsApp alerts for critical failures"""
    
    def __init__(self, account_sid: str = None, auth_token: str = None, from_number: str = None):
        """
        Initialize WhatsApp alerter
        
        Args:
            account_sid: Twilio account SID (or set TWILIO_ACCOUNT_SID env var)
            auth_token: Twilio auth token (or set TWILIO_AUTH_TOKEN env var)
            from_number: Twilio WhatsApp number (or set TWILIO_WHATSAPP_NUMBER env var)
        """
        self.account_sid = account_sid or os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = auth_token or os.getenv('TWILIO_AUTH_TOKEN')
        self.from_number = from_number or os.getenv('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')
        
        if not self.account_sid or not self.auth_token:
            logger.warning("WhatsApp alerts not configured. Set TWILIO credentials.")
            self.client = None
        else:
            try:
                self.client = Client(self.account_sid, self.auth_token)
                logger.info("✓ WhatsApp alerter initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Twilio client: {e}")
                self.client = None
    
    def send_alert(self, 
                   to_number: str, 
                   machine_id: str,
                   failure_type: str,
                   probability: float,
                   time_to_failure: Optional[int] = None,
                   language: str = 'ar') -> bool:
        """
        Send WhatsApp alert
        
        Args:
            to_number: Recipient WhatsApp number (format: whatsapp:+213XXXXXXXXX for Algeria)
            machine_id: Machine identifier
            failure_type: Type of predicted failure
            probability: Failure probability (0-1)
            time_to_failure: Estimated hours until failure
            language: Message language ('ar' or 'fr')
            
        Returns:
            True if sent successfully
        """
        if not self.client:
            logger.error("WhatsApp client not initialized")
            return False
        
        # Ensure number has whatsapp: prefix
        if not to_number.startswith('whatsapp:'):
            to_number = f'whatsapp:{to_number}'
        
        # Build message based on language
        message = self._build_message(
            machine_id, failure_type, probability, 
            time_to_failure, language
        )
        
        try:
            sent_message = self.client.messages.create(
                from_=self.from_number,
                body=message,
                to=to_number
            )
            
            logger.info(f"✓ WhatsApp alert sent to {to_number}: {sent_message.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send WhatsApp alert: {e}")
            return False
    
    def _build_message(self, 
                       machine_id: str,
                       failure_type: str,
                       probability: float,
                       time_to_failure: Optional[int],
                       language: str) -> str:
        """Build alert message in specified language"""
        
        # Risk emoji based on probability
        if probability >= 0.8:
            emoji = "🔴"
        elif probability >= 0.5:
            emoji = "🟡"
        else:
            emoji = "🟢"
        
        if language == 'ar':
            message = f"""
{emoji} *تحذير صيانة عاجلة*

🔧 الآلة: {machine_id}
⚠️ نوع العطل المتوقع: {failure_type}
📊 احتمالية العطل: {probability:.0%}
"""
            if time_to_failure:
                if time_to_failure < 24:
                    message += f"⏰ الوقت المتبقي: ~{time_to_failure} ساعة\n"
                else:
                    days = time_to_failure // 24
                    message += f"⏰ الوقت المتبقي: ~{days} يوم\n"
            
            message += f"\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            message += "\n\n✅ يرجى اتخاذ الإجراءات اللازمة فورًا"
            
        else:  # French
            message = f"""
{emoji} *Alerte Maintenance Urgente*

🔧 Machine: {machine_id}
⚠️ Type de panne prévue: {failure_type}
📊 Probabilité: {probability:.0%}
"""
            if time_to_failure:
                if time_to_failure < 24:
                    message += f"⏰ Temps restant: ~{time_to_failure}h\n"
                else:
                    days = time_to_failure // 24
                    message += f"⏰ Temps restant: ~{days} jours\n"
            
            message += f"\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            message += "\n\n✅ Action immédiate requise"
        
        return message
    
    def send_daily_summary(self,
                          to_number: str,
                          machines_at_risk: int,
                          critical_count: int,
                          language: str = 'ar') -> bool:
        """
        Send daily summary of system status
        
        Args:
            to_number: Recipient WhatsApp number
            machines_at_risk: Number of machines at risk
            critical_count: Number of critical alerts
            language: Message language
            
        Returns:
            True if sent successfully
        """
        if not self.client:
            logger.error("WhatsApp client not initialized")
            return False
        
        if not to_number.startswith('whatsapp:'):
            to_number = f'whatsapp:{to_number}'
        
        if language == 'ar':
            message = f"""
📊 *ملخص يومي - نظام الصيانة التنبؤية*

📅 {datetime.now().strftime('%Y-%m-%d')}

🔴 تنبيهات عاجلة: {critical_count}
⚠️ آلات في خطر: {machines_at_risk}

✅ تحقق من النظام للحصول على التفاصيل
"""
        else:
            message = f"""
📊 *Résumé Quotidien - Maintenance Prédictive*

📅 {datetime.now().strftime('%Y-%m-%d')}

🔴 Alertes critiques: {critical_count}
⚠️ Machines à risque: {machines_at_risk}

✅ Consultez le système pour plus de détails
"""
        
        try:
            sent_message = self.client.messages.create(
                from_=self.from_number,
                body=message,
                to=to_number
            )
            
            logger.info(f"✓ Daily summary sent to {to_number}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")
            return False
    
    def test_connection(self, to_number: str) -> bool:
        """
        Test WhatsApp connection with a simple message
        
        Args:
            to_number: Test recipient number
            
        Returns:
            True if successful
        """
        if not self.client:
            logger.error("WhatsApp client not initialized")
            return False
        
        if not to_number.startswith('whatsapp:'):
            to_number = f'whatsapp:{to_number}'
        
        try:
            message = self.client.messages.create(
                from_=self.from_number,
                body="✅ WhatsApp alerts configured successfully!\n\nصيانة تنبؤية - Maintenance Prédictive",
                to=to_number
            )
            
            logger.info(f"✓ Test message sent: {message.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Test message failed: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize (requires Twilio credentials)
    alerter = WhatsAppAlerter()
    
    # Example: Send alert
    # alerter.send_alert(
    #     to_number='whatsapp:+213XXXXXXXXX',  # Algerian number
    #     machine_id='M001',
    #     failure_type='Heat Dissipation Failure',
    #     probability=0.85,
    #     time_to_failure=12,
    #     language='ar'
    # )
    
    logger.info("WhatsApp alerter ready")
