"""
Alert Manager for Anomaly Detection System

Handles notifications through multiple channels with rate limiting and severity levels.
"""

import json
import os
import smtplib
import requests
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional, List
import logging
from collections import defaultdict
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages alerts for anomaly detection with multiple notification channels.
    """
    
    def __init__(self, config_path: str = "alert_config.json"):
        """
        Initialize Alert Manager with configuration.
        
        Parameters:
        -----------
        config_path: str
            Path to JSON configuration file
        """
        self.config = self._load_config(config_path)
        self.alert_history = defaultdict(list)
        self.cooldown_tracker = {}
        self.alert_log_file = "alerts.log"
        
        # Severity thresholds
        self.severity_rules = {
            'critical': {
                'z_score_threshold': 5.0,
                'price_change_threshold': 0.05,  # 5%
                'volume_spike_threshold': 10.0,
                'cooldown_minutes': 5
            },
            'high': {
                'z_score_threshold': 4.0,
                'price_change_threshold': 0.03,  # 3%
                'volume_spike_threshold': 5.0,
                'cooldown_minutes': 15
            },
            'medium': {
                'z_score_threshold': 3.0,
                'price_change_threshold': 0.02,  # 2%
                'volume_spike_threshold': 3.0,
                'cooldown_minutes': 30
            },
            'low': {
                'z_score_threshold': 2.5,
                'price_change_threshold': 0.015,  # 1.5%
                'volume_spike_threshold': 2.0,
                'cooldown_minutes': 60
            }
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file or use defaults"""
        default_config = {
            "telegram": {
                "enabled": False,
                "token": os.getenv("TELEGRAM_TOKEN", ""),
                "chat_id": os.getenv("TELEGRAM_CHAT_ID", "")
            },
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "from_email": os.getenv("ALERT_EMAIL", ""),
                "from_password": os.getenv("ALERT_EMAIL_PASSWORD", ""),
                "to_emails": []
            },
            "slack": {
                "enabled": False,
                "webhook_url": os.getenv("SLACK_WEBHOOK_URL", "")
            },
            "discord": {
                "enabled": False,
                "webhook_url": os.getenv("DISCORD_WEBHOOK_URL", "")
            },
            "console": {
                "enabled": True
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key in loaded_config:
                        default_config[key].update(loaded_config[key])
        else:
            # Save default config for reference
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
                
        return default_config
    
    def determine_severity(self, anomaly_data: Dict) -> str:
        """
        Determine alert severity based on anomaly characteristics.
        
        Parameters:
        -----------
        anomaly_data: Dict
            Dictionary containing anomaly details
            
        Returns:
        --------
        str: Severity level ('critical', 'high', 'medium', 'low')
        """
        z_score = abs(anomaly_data.get('z_score', 0))
        price_change = abs(anomaly_data.get('price_change_pct', 0)) / 100
        volume_spike = anomaly_data.get('volume_spike', 1)
        
        # Check from highest to lowest severity
        for severity, rules in self.severity_rules.items():
            if (z_score >= rules['z_score_threshold'] or
                price_change >= rules['price_change_threshold'] or
                volume_spike >= rules['volume_spike_threshold']):
                
                # Additional checks for critical
                if severity == 'critical':
                    # Require at least 2 conditions for critical
                    conditions_met = sum([
                        z_score >= rules['z_score_threshold'],
                        price_change >= rules['price_change_threshold'],
                        volume_spike >= rules['volume_spike_threshold']
                    ])
                    if conditions_met >= 2:
                        return 'critical'
                else:
                    return severity
                    
        return 'low'
    
    def send_alert(self, anomaly_data: Dict, severity: Optional[str] = None):
        """
        Send alert through configured channels based on severity.
        
        Parameters:
        -----------
        anomaly_data: Dict
            Anomaly information including price, z_score, timestamp, etc.
        severity: str, optional
            Override automatic severity determination
        """
        # Determine severity if not provided
        if severity is None:
            severity = self.determine_severity(anomaly_data)
        
        # Check cooldown
        cooldown_key = f"{anomaly_data.get('anomaly_type', 'unknown')}_{severity}"
        if self._in_cooldown(cooldown_key, severity):
            logger.info(f"Alert suppressed due to cooldown: {cooldown_key}")
            return
        
        # Format alert message
        message = self._format_alert_message(anomaly_data, severity)
        
        # Log alert
        self._log_alert(anomaly_data, severity, message)
        
        # Send based on severity
        channels_used = []
        
        if severity == 'critical':
            # Send to all available channels
            channels_used.extend(self._send_telegram(message, anomaly_data))
            channels_used.extend(self._send_email(message, anomaly_data, severity))
            channels_used.extend(self._send_slack(message, anomaly_data, severity))
            channels_used.extend(self._send_discord(message, anomaly_data, severity))
            
        elif severity == 'high':
            # Send to instant channels
            channels_used.extend(self._send_telegram(message, anomaly_data))
            channels_used.extend(self._send_slack(message, anomaly_data, severity))
            channels_used.extend(self._send_discord(message, anomaly_data, severity))
            
        elif severity == 'medium':
            # Send to one instant channel
            channels_used.extend(self._send_telegram(message, anomaly_data))
            
        # Always log to console if enabled
        if self.config['console']['enabled']:
            self._send_console(message, severity)
            channels_used.append('console')
        
        # Update cooldown
        self._update_cooldown(cooldown_key, severity)
        
        # Record in history
        self.alert_history[severity].append({
            'timestamp': datetime.now(),
            'anomaly_data': anomaly_data,
            'channels': channels_used
        })
        
    def _format_alert_message(self, anomaly_data: Dict, severity: str) -> str:
        """Format alert message for sending"""
        emoji_map = {
            'critical': '🚨🔴',
            'high': '⚠️🟠',
            'medium': '⚡🟡',
            'low': 'ℹ️🔵'
        }
        
        message = f"{emoji_map.get(severity, '📢')} **{severity.upper()} ANOMALY ALERT**\n\n"
        message += f"**Type:** {anomaly_data.get('anomaly_type', 'Unknown')}\n"
        message += f"**Time:** {anomaly_data.get('timestamp', 'N/A')}\n"
        message += f"**Price:** ${anomaly_data.get('price', 0):,.2f}\n"
        message += f"**Z-Score:** {anomaly_data.get('z_score', 0):.2f}\n"
        
        if 'price_change_pct' in anomaly_data:
            message += f"**Price Change:** {anomaly_data['price_change_pct']:.2f}%\n"
        
        if 'volume_spike' in anomaly_data:
            message += f"**Volume Spike:** {anomaly_data['volume_spike']:.1f}x normal\n"
        
        if 'vwap_deviation' in anomaly_data:
            message += f"**VWAP Deviation:** {anomaly_data['vwap_deviation']:.2f}%\n"
        
        # Add model votes if available
        if 'model_votes' in anomaly_data:
            votes = json.loads(anomaly_data['model_votes']) if isinstance(anomaly_data['model_votes'], str) else anomaly_data['model_votes']
            anomaly_models = [k for k, v in votes.items() if v == -1]
            if anomaly_models:
                message += f"**ML Models:** {', '.join(anomaly_models)}\n"
        
        return message
    
    def _send_telegram(self, message: str, anomaly_data: Dict) -> List[str]:
        """Send Telegram notification"""
        if not self.config['telegram']['enabled']:
            return []
            
        try:
            # Convert markdown to Telegram format
            telegram_message = message.replace('**', '*')  # Bold in Telegram
            
            response = requests.post(
                f"https://api.telegram.org/bot{self.config['telegram']['token']}/sendMessage",
                data={
                    'chat_id': self.config['telegram']['chat_id'],
                    'text': telegram_message,
                    'parse_mode': 'Markdown'
                }
            )
            
            if response.status_code == 200:
                logger.info("Telegram alert sent successfully")
                return ['telegram']
            else:
                logger.error(f"Failed to send Telegram alert: {response.text}")
                
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
            
        return []
    
    def _send_email(self, message: str, anomaly_data: Dict, severity: str) -> List[str]:
        """Send email notification"""
        if not self.config['email']['enabled'] or not self.config['email']['to_emails']:
            return []
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['email']['from_email']
            msg['To'] = ', '.join(self.config['email']['to_emails'])
            msg['Subject'] = f"{severity.upper()} Crypto Anomaly Alert - {anomaly_data.get('anomaly_type', 'Unknown')}"
            
            # Convert markdown to HTML for email
            html_message = message.replace('**', '<b>').replace('**', '</b>')
            html_message = html_message.replace('\n', '<br>')
            
            msg.attach(MIMEText(html_message, 'html'))
            
            with smtplib.SMTP(self.config['email']['smtp_server'], self.config['email']['smtp_port']) as server:
                server.starttls()
                server.login(self.config['email']['from_email'], self.config['email']['from_password'])
                server.send_message(msg)
                
            logger.info("Email alert sent successfully")
            return ['email']
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            
        return []
    
    def _send_slack(self, message: str, anomaly_data: Dict, severity: str) -> List[str]:
        """Send Slack notification"""
        if not self.config['slack']['enabled']:
            return []
            
        try:
            # Format for Slack
            slack_data = {
                "text": f"{severity.upper()} Anomaly Alert",
                "attachments": [{
                    "color": {
                        'critical': 'danger',
                        'high': 'warning', 
                        'medium': '#ff9800',
                        'low': 'good'
                    }.get(severity, '#888888'),
                    "fields": [
                        {"title": "Type", "value": anomaly_data.get('anomaly_type', 'Unknown'), "short": True},
                        {"title": "Price", "value": f"${anomaly_data.get('price', 0):,.2f}", "short": True},
                        {"title": "Z-Score", "value": f"{anomaly_data.get('z_score', 0):.2f}", "short": True},
                        {"title": "Time", "value": anomaly_data.get('timestamp', 'N/A'), "short": True}
                    ],
                    "footer": "Anomaly Detection System",
                    "ts": int(datetime.now().timestamp())
                }]
            }
            
            response = requests.post(self.config['slack']['webhook_url'], json=slack_data)
            
            if response.status_code == 200:
                logger.info("Slack alert sent successfully")
                return ['slack']
            else:
                logger.error(f"Failed to send Slack alert: {response.text}")
                
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            
        return []
    
    def _send_discord(self, message: str, anomaly_data: Dict, severity: str) -> List[str]:
        """Send Discord notification"""
        if not self.config['discord']['enabled']:
            return []
            
        try:
            # Format for Discord
            color_map = {
                'critical': 0xFF0000,  # Red
                'high': 0xFF6600,      # Orange
                'medium': 0xFFCC00,    # Yellow
                'low': 0x0099FF        # Blue
            }
            
            discord_data = {
                "embeds": [{
                    "title": f"{severity.upper()} Anomaly Alert",
                    "color": color_map.get(severity, 0x888888),
                    "fields": [
                        {"name": "Type", "value": anomaly_data.get('anomaly_type', 'Unknown'), "inline": True},
                        {"name": "Price", "value": f"${anomaly_data.get('price', 0):,.2f}", "inline": True},
                        {"name": "Z-Score", "value": f"{anomaly_data.get('z_score', 0):.2f}", "inline": True},
                        {"name": "Volume Spike", "value": f"{anomaly_data.get('volume_spike', 1):.1f}x", "inline": True},
                        {"name": "Price Change", "value": f"{anomaly_data.get('price_change_pct', 0):.2f}%", "inline": True},
                        {"name": "Time", "value": anomaly_data.get('timestamp', 'N/A'), "inline": True}
                    ],
                    "footer": {"text": "Crypto Anomaly Detection"},
                    "timestamp": datetime.now().isoformat()
                }]
            }
            
            response = requests.post(self.config['discord']['webhook_url'], json=discord_data)
            
            if response.status_code in [200, 204]:
                logger.info("Discord alert sent successfully")
                return ['discord']
            else:
                logger.error(f"Failed to send Discord alert: {response.text}")
                
        except Exception as e:
            logger.error(f"Error sending Discord alert: {e}")
            
        return []
    
    def _send_console(self, message: str, severity: str):
        """Print alert to console with formatting"""
        color_map = {
            'critical': '\033[91m',  # Red
            'high': '\033[93m',      # Yellow
            'medium': '\033[94m',    # Blue
            'low': '\033[92m'        # Green
        }
        
        color = color_map.get(severity, '\033[0m')
        reset = '\033[0m'
        
        print(f"\n{color}{'='*60}")
        print(message)
        print(f"{'='*60}{reset}\n")
    
    def _in_cooldown(self, key: str, severity: str) -> bool:
        """Check if alert is in cooldown period"""
        if key not in self.cooldown_tracker:
            return False
            
        last_alert_time = self.cooldown_tracker[key]
        cooldown_minutes = self.severity_rules[severity]['cooldown_minutes']
        cooldown_delta = timedelta(minutes=cooldown_minutes)
        
        return datetime.now() - last_alert_time < cooldown_delta
    
    def _update_cooldown(self, key: str, severity: str):
        """Update cooldown tracker"""
        self.cooldown_tracker[key] = datetime.now()
    
    def _log_alert(self, anomaly_data: Dict, severity: str, message: str):
        """Log alert to file"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'anomaly_type': anomaly_data.get('anomaly_type', 'unknown'),
            'price': anomaly_data.get('price', 0),
            'z_score': anomaly_data.get('z_score', 0),
            'price_change_pct': anomaly_data.get('price_change_pct', 0),
            'volume_spike': anomaly_data.get('volume_spike', 1),
            'vwap_deviation': anomaly_data.get('vwap_deviation', 0),
            'model_votes': anomaly_data.get('model_votes', {}),
            'message': message
        }

        try:
            with open(self.alert_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write alert to log file: {e}")
