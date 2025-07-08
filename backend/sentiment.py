"""
Sentiment Analysis Module - News and social media sentiment for portfolio holdings
"""

import requests
import pandas as pd
import numpy as np
from textblob import TextBlob
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import time

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analyze sentiment from news and social media for portfolio stocks
    """
    
    def __init__(self):
        # In a production environment, you would use actual API keys
        self.news_api_key = "your_news_api_key_here"  # Replace with actual API key
        self.alpha_vantage_key = "your_alpha_vantage_key_here"  # Replace with actual API key
        
        # Sentiment scoring weights
        self.sentiment_weights = {
            'news': 0.6,
            'social': 0.3,
            'technical': 0.1
        }
    
    def analyze_portfolio_sentiment(self, symbols: List[str]) -> Dict:
        """
        Analyze sentiment for entire portfolio
        """
        try:
            portfolio_sentiment = {}
            overall_scores = []
            
            for symbol in symbols:
                sentiment_data = self.analyze_stock_sentiment(symbol)
                portfolio_sentiment[symbol] = sentiment_data
                
                if sentiment_data.get('overall_score') is not None:
                    overall_scores.append(sentiment_data['overall_score'])
            
            # Calculate portfolio-level sentiment
            portfolio_score = np.mean(overall_scores) if overall_scores else 0.0
            
            return {
                'individual_sentiment': portfolio_sentiment,
                'portfolio_sentiment': {
                    'overall_score': float(portfolio_score),
                    'sentiment_label': self._get_sentiment_label(portfolio_score),
                    'confidence': self._calculate_confidence(overall_scores),
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'sentiment_distribution': self._calculate_sentiment_distribution(portfolio_sentiment),
                'risk_indicators': self._identify_sentiment_risks(portfolio_sentiment)
            }
            
        except Exception as e:
            logger.error(f"Error in portfolio sentiment analysis: {e}")
            return self._get_default_sentiment_response()
    
    def analyze_stock_sentiment(self, symbol: str) -> Dict:
        """
        Analyze sentiment for a single stock
        """
        try:
            # Get news sentiment
            news_sentiment = self._get_news_sentiment(symbol)
            
            # Get social media sentiment (simulated)
            social_sentiment = self._get_social_sentiment(symbol)
            
            # Get technical sentiment indicators
            technical_sentiment = self._get_technical_sentiment(symbol)
            
            # Calculate weighted overall sentiment
            overall_score = (
                news_sentiment['score'] * self.sentiment_weights['news'] +
                social_sentiment['score'] * self.sentiment_weights['social'] +
                technical_sentiment['score'] * self.sentiment_weights['technical']
            )
            
            return {
                'symbol': symbol,
                'overall_score': float(overall_score),
                'sentiment_label': self._get_sentiment_label(overall_score),
                'components': {
                    'news': news_sentiment,
                    'social': social_sentiment,
                    'technical': technical_sentiment
                },
                'analysis_timestamp': datetime.now().isoformat(),
                'confidence': self._calculate_component_confidence([
                    news_sentiment, social_sentiment, technical_sentiment
                ])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return self._get_default_stock_sentiment(symbol)
    
    def _get_news_sentiment(self, symbol: str) -> Dict:
        """
        Get sentiment from financial news
        """
        try:
            # Simulate news sentiment analysis
            # In production, you would use actual news APIs like NewsAPI, Alpha Vantage News, etc.
            
            # Mock news headlines for demonstration
            mock_headlines = [
                f"{symbol} reports strong quarterly earnings",
                f"Analysts upgrade {symbol} price target",
                f"{symbol} announces new product launch",
                f"Market volatility affects {symbol} trading",
                f"{symbol} CEO discusses future growth plans"
            ]
            
            sentiments = []
            for headline in mock_headlines:
                blob = TextBlob(headline)
                sentiments.append(blob.sentiment.polarity)
            
            avg_sentiment = np.mean(sentiments)
            
            return {
                'score': float(avg_sentiment),
                'articles_analyzed': len(mock_headlines),
                'source': 'financial_news',
                'confidence': min(0.8, len(mock_headlines) / 10),  # Higher confidence with more articles
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Error getting news sentiment for {symbol}: {e}")
            return {'score': 0.0, 'articles_analyzed': 0, 'source': 'financial_news', 'confidence': 0.0}
    
    def _get_social_sentiment(self, symbol: str) -> Dict:
        """
        Get sentiment from social media (simulated)
        """
        try:
            # Simulate social media sentiment
            # In production, you would use Twitter API, Reddit API, etc.
            
            # Generate mock social sentiment based on symbol characteristics
            base_sentiment = np.random.normal(0, 0.3)  # Random sentiment around neutral
            
            # Add some symbol-specific bias (this is just for demonstration)
            if symbol in ['AAPL', 'GOOGL', 'MSFT']:
                base_sentiment += 0.1  # Tech stocks tend to have positive sentiment
            elif symbol in ['XOM', 'CVX']:
                base_sentiment -= 0.05  # Energy stocks might have mixed sentiment
            
            # Clamp to [-1, 1] range
            sentiment_score = np.clip(base_sentiment, -1, 1)
            
            return {
                'score': float(sentiment_score),
                'mentions_analyzed': np.random.randint(50, 500),
                'source': 'social_media',
                'confidence': 0.6,  # Social media sentiment is generally less reliable
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Error getting social sentiment for {symbol}: {e}")
            return {'score': 0.0, 'mentions_analyzed': 0, 'source': 'social_media', 'confidence': 0.0}
    
    def _get_technical_sentiment(self, symbol: str) -> Dict:
        """
        Get technical sentiment indicators
        """
        try:
            # Simulate technical sentiment based on price action
            # In production, you would calculate actual technical indicators
            
            # Mock technical indicators
            rsi = np.random.uniform(30, 70)  # RSI between 30-70
            macd_signal = np.random.choice([-1, 0, 1])  # MACD signal
            
            # Convert technical indicators to sentiment score
            rsi_sentiment = (rsi - 50) / 50  # Normalize RSI to [-1, 1]
            macd_sentiment = macd_signal * 0.3
            
            technical_score = (rsi_sentiment + macd_sentiment) / 2
            
            return {
                'score': float(np.clip(technical_score, -1, 1)),
                'indicators': {
                    'rsi': float(rsi),
                    'macd_signal': int(macd_signal),
                    'trend': 'bullish' if technical_score > 0.1 else 'bearish' if technical_score < -0.1 else 'neutral'
                },
                'source': 'technical_analysis',
                'confidence': 0.7,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Error getting technical sentiment for {symbol}: {e}")
            return {'score': 0.0, 'indicators': {}, 'source': 'technical_analysis', 'confidence': 0.0}
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to human-readable label"""
        if score > 0.3:
            return "Very Positive"
        elif score > 0.1:
            return "Positive"
        elif score > -0.1:
            return "Neutral"
        elif score > -0.3:
            return "Negative"
        else:
            return "Very Negative"
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calculate confidence based on score consistency"""
        if not scores:
            return 0.0
        
        # Higher confidence when scores are consistent
        std_dev = np.std(scores)
        confidence = max(0.0, 1.0 - std_dev)
        return float(confidence)
    
    def _calculate_component_confidence(self, components: List[Dict]) -> float:
        """Calculate confidence based on component reliabilities"""
        confidences = [comp.get('confidence', 0.0) for comp in components]
        return float(np.mean(confidences))
    
    def _calculate_sentiment_distribution(self, portfolio_sentiment: Dict) -> Dict:
        """Calculate distribution of sentiment across portfolio"""
        labels = []
        for symbol_data in portfolio_sentiment.values():
            if isinstance(symbol_data, dict) and 'sentiment_label' in symbol_data:
                labels.append(symbol_data['sentiment_label'])
        
        if not labels:
            return {}
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        distribution = {}
        for label, count in zip(unique_labels, counts):
            distribution[label] = {
                'count': int(count),
                'percentage': float(count / total * 100)
            }
        
        return distribution
    
    def _identify_sentiment_risks(self, portfolio_sentiment: Dict) -> List[Dict]:
        """Identify potential risks based on sentiment analysis"""
        risks = []
        
        for symbol, data in portfolio_sentiment.items():
            if isinstance(data, dict) and 'overall_score' in data:
                score = data['overall_score']
                
                if score < -0.3:
                    risks.append({
                        'symbol': symbol,
                        'risk_type': 'negative_sentiment',
                        'severity': 'high' if score < -0.5 else 'medium',
                        'description': f"{symbol} shows very negative sentiment ({score:.2f})"
                    })
                elif data.get('confidence', 0) < 0.3:
                    risks.append({
                        'symbol': symbol,
                        'risk_type': 'low_confidence',
                        'severity': 'low',
                        'description': f"Low confidence in sentiment analysis for {symbol}"
                    })
        
        return risks
    
    def _get_default_sentiment_response(self) -> Dict:
        """Return default sentiment response when analysis fails"""
        return {
            'individual_sentiment': {},
            'portfolio_sentiment': {
                'overall_score': 0.0,
                'sentiment_label': 'Neutral',
                'confidence': 0.0,
                'analysis_timestamp': datetime.now().isoformat()
            },
            'sentiment_distribution': {},
            'risk_indicators': []
        }
    
    def _get_default_stock_sentiment(self, symbol: str) -> Dict:
        """Return default sentiment for a single stock when analysis fails"""
        return {
            'symbol': symbol,
            'overall_score': 0.0,
            'sentiment_label': 'Neutral',
            'components': {
                'news': {'score': 0.0, 'confidence': 0.0},
                'social': {'score': 0.0, 'confidence': 0.0},
                'technical': {'score': 0.0, 'confidence': 0.0}
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'confidence': 0.0
        }
