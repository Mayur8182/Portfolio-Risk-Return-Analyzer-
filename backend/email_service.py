"""
Email Service for Portfolio Analytics Platform
Handles PDF report delivery via email
"""

import os
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from typing import Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailService:
    """Professional email service for PDF report delivery"""
    
    def __init__(self):
        """Initialize email service with environment variables"""
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.username = os.getenv('MAIL_USERNAME') or os.getenv('SMTP_USERNAME')
        self.password = os.getenv('MAIL_PASSWORD') or os.getenv('SMTP_PASSWORD')
        self.enabled = os.getenv('ENABLE_EMAIL_REPORTS', 'true').lower() == 'true'
        
        if not self.username or not self.password:
            logger.warning("⚠️ Email credentials not configured. Email reports disabled.")
            self.enabled = False
        else:
            logger.info(f"✅ Email service initialized: {self.username}")
    
    def send_portfolio_report(self, 
                            recipient_email: str, 
                            pdf_path: str, 
                            portfolio_data: dict,
                            analysis_summary: dict) -> bool:
        """
        Send portfolio analysis report via email
        
        Args:
            recipient_email: Email address to send report to
            pdf_path: Path to the PDF report file
            portfolio_data: Portfolio configuration data
            analysis_summary: Summary of analysis results
            
        Returns:
            bool: True if email sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning("📧 Email service disabled. Cannot send report.")
            return False
            
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = recipient_email
            msg['Subject'] = self._create_subject(portfolio_data)
            
            # Create email body
            body = self._create_email_body(portfolio_data, analysis_summary)
            msg.attach(MIMEText(body, 'html'))
            
            # Attach PDF report
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(pdf_path)}'
                )
                msg.attach(part)
            else:
                logger.error(f"❌ PDF file not found: {pdf_path}")
                return False
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            text = msg.as_string()
            server.sendmail(self.username, recipient_email, text)
            server.quit()
            
            logger.info(f"✅ Portfolio report sent successfully to {recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send email: {str(e)}")
            return False
    
    def _create_subject(self, portfolio_data: dict) -> str:
        """Create email subject line"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        portfolio_name = portfolio_data.get('name', 'Portfolio')
        return f"📊 {portfolio_name} Analysis Report - {date_str}"
    
    def _create_email_body(self, portfolio_data: dict, analysis_summary: dict) -> str:
        """Create HTML email body"""
        stocks = portfolio_data.get('stocks', [])
        weights = portfolio_data.get('weights', [])
        
        # Portfolio composition
        portfolio_composition = ""
        for i, stock in enumerate(stocks):
            weight = weights[i] if i < len(weights) else 0
            portfolio_composition += f"<li><strong>{stock}</strong>: {weight*100:.1f}%</li>"
        
        # Key metrics
        performance = analysis_summary.get('performance_metrics', {})
        risk = analysis_summary.get('risk_metrics', {})
        
        total_return = performance.get('portfolio_total_return', 0) * 100
        sharpe_ratio = risk.get('sharpe_ratio', 0)
        max_drawdown = risk.get('max_drawdown', 0) * 100
        volatility = risk.get('annualized_volatility', 0) * 100
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .metrics {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0; }}
                .metric-item {{ display: inline-block; margin: 10px 15px; }}
                .metric-value {{ font-size: 1.2em; font-weight: bold; color: #2c3e50; }}
                .portfolio-list {{ background: #e8f4fd; padding: 15px; border-radius: 8px; }}
                .footer {{ background: #34495e; color: white; padding: 15px; text-align: center; margin-top: 20px; }}
                .positive {{ color: #27ae60; }}
                .negative {{ color: #e74c3c; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Portfolio Analysis Report</h1>
                <p>Professional Financial Analytics Platform</p>
                <p>{datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
            </div>
            
            <div class="content">
                <h2>🎯 Executive Summary</h2>
                <p>Your portfolio analysis has been completed successfully. Please find the detailed PDF report attached to this email.</p>
                
                <div class="metrics">
                    <h3>📈 Key Performance Metrics</h3>
                    <div class="metric-item">
                        <div>Total Return</div>
                        <div class="metric-value {'positive' if total_return >= 0 else 'negative'}">{total_return:+.2f}%</div>
                    </div>
                    <div class="metric-item">
                        <div>Sharpe Ratio</div>
                        <div class="metric-value">{sharpe_ratio:.3f}</div>
                    </div>
                    <div class="metric-item">
                        <div>Max Drawdown</div>
                        <div class="metric-value negative">{max_drawdown:.2f}%</div>
                    </div>
                    <div class="metric-item">
                        <div>Volatility</div>
                        <div class="metric-value">{volatility:.2f}%</div>
                    </div>
                </div>
                
                <div class="portfolio-list">
                    <h3>💼 Portfolio Composition</h3>
                    <ul>
                        {portfolio_composition}
                    </ul>
                </div>
                
                <h3>📋 Report Contents</h3>
                <p>The attached PDF report includes:</p>
                <ul>
                    <li>📊 Comprehensive performance analysis</li>
                    <li>⚠️ Risk assessment and metrics</li>
                    <li>📈 Interactive charts and visualizations</li>
                    <li>🎯 Portfolio optimization recommendations</li>
                    <li>📰 Market sentiment analysis</li>
                    <li>🔮 Monte Carlo simulations</li>
                </ul>
                
                <h3>🔒 Security & Privacy</h3>
                <p>This report contains confidential financial information. Please handle with appropriate care and do not share with unauthorized parties.</p>
            </div>
            
            <div class="footer">
                <p><strong>Portfolio Analytics Platform</strong></p>
                <p>Professional-grade financial analysis powered by institutional data sources</p>
                <p>📧 Generated automatically by our secure email system</p>
            </div>
        </body>
        </html>
        """
        
        return html_body
    
    def test_connection(self) -> bool:
        """Test email service connection"""
        if not self.enabled:
            return False
            
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.quit()
            logger.info("✅ Email service connection test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Email service connection test failed: {str(e)}")
            return False

# Global email service instance
email_service = EmailService()
