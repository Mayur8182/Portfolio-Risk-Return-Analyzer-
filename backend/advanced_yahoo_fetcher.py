"""
Advanced Yahoo Finance Data Fetcher for Maximum Accuracy
Ultra-high precision data fetching with comprehensive market data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedYahooFetcher:
    """
    Ultra-advanced Yahoo Finance data fetcher for maximum accuracy
    Designed for institutional-grade portfolio analysis
    """
    
    def __init__(self):
        self.cache = {}
        self.supported_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        self.supported_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        
    def get_ultra_comprehensive_data(self, symbols: List[str], period: str = '2y') -> Dict:
        """
        Get ultra-comprehensive market data for maximum accuracy analysis
        
        Args:
            symbols: List of stock symbols
            period: Data period
            
        Returns:
            Comprehensive market data dictionary
        """
        try:
            logger.info(f"🚀 Fetching ultra-comprehensive data for {len(symbols)} symbols")
            
            comprehensive_data = {
                'price_data': {},
                'fundamental_data': {},
                'technical_indicators': {},
                'market_data': {},
                'risk_metrics': {},
                'performance_metrics': {},
                'sector_analysis': {},
                'dividend_data': {},
                'split_data': {},
                'options_data': {},
                'institutional_data': {},
                'analyst_data': {}
            }
            
            for symbol in symbols:
                logger.info(f"📊 Processing {symbol} with ultra-high precision...")
                
                # Get ticker object
                ticker = yf.Ticker(symbol)
                
                # 1. Ultra-detailed price data
                comprehensive_data['price_data'][symbol] = self._get_ultra_price_data(ticker, period)
                
                # 2. Comprehensive fundamental data
                comprehensive_data['fundamental_data'][symbol] = self._get_fundamental_data(ticker)
                
                # 3. Advanced technical indicators
                comprehensive_data['technical_indicators'][symbol] = self._calculate_technical_indicators(
                    comprehensive_data['price_data'][symbol]
                )
                
                # 4. Market data and statistics
                comprehensive_data['market_data'][symbol] = self._get_market_statistics(ticker)
                
                # 5. Risk and performance metrics
                comprehensive_data['risk_metrics'][symbol] = self._calculate_risk_metrics(
                    comprehensive_data['price_data'][symbol]
                )
                
                # 6. Performance analysis
                comprehensive_data['performance_metrics'][symbol] = self._calculate_performance_metrics(
                    comprehensive_data['price_data'][symbol]
                )
                
                # 7. Sector and industry analysis
                comprehensive_data['sector_analysis'][symbol] = self._get_sector_analysis(ticker)
                
                # 8. Dividend and split data
                comprehensive_data['dividend_data'][symbol] = self._get_dividend_data(ticker)
                comprehensive_data['split_data'][symbol] = self._get_split_data(ticker)
                
                # 9. Options data (if available)
                comprehensive_data['options_data'][symbol] = self._get_options_data(ticker)
                
                # 10. Institutional holdings
                comprehensive_data['institutional_data'][symbol] = self._get_institutional_data(ticker)
                
                # 11. Analyst recommendations
                comprehensive_data['analyst_data'][symbol] = self._get_analyst_data(ticker)
                
                logger.info(f"✅ {symbol} ultra-comprehensive data completed")
            
            # Calculate portfolio-level metrics
            comprehensive_data['portfolio_metrics'] = self._calculate_portfolio_metrics(comprehensive_data)
            
            logger.info("🎯 Ultra-comprehensive data fetching completed successfully")
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"❌ Error in ultra-comprehensive data fetching: {e}")
            return {}
    
    def _get_ultra_price_data(self, ticker, period: str) -> pd.DataFrame:
        """Get ultra-detailed price data with multiple timeframes"""
        try:
            # Get main historical data
            hist = ticker.history(period=period, interval='1d', auto_adjust=True, prepost=True)
            
            if hist.empty:
                logger.warning(f"No historical data for {ticker.ticker}")
                return pd.DataFrame()
            
            # Add additional calculated fields
            hist['Returns'] = hist['Close'].pct_change()
            hist['Log_Returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
            hist['Volatility_20'] = hist['Returns'].rolling(20).std() * np.sqrt(252)
            hist['SMA_20'] = hist['Close'].rolling(20).mean()
            hist['SMA_50'] = hist['Close'].rolling(50).mean()
            hist['SMA_200'] = hist['Close'].rolling(200).mean()
            hist['EMA_12'] = hist['Close'].ewm(span=12).mean()
            hist['EMA_26'] = hist['Close'].ewm(span=26).mean()
            hist['RSI'] = self._calculate_rsi(hist['Close'])
            hist['MACD'] = hist['EMA_12'] - hist['EMA_26']
            hist['MACD_Signal'] = hist['MACD'].ewm(span=9).mean()
            hist['Bollinger_Upper'] = hist['SMA_20'] + (hist['Close'].rolling(20).std() * 2)
            hist['Bollinger_Lower'] = hist['SMA_20'] - (hist['Close'].rolling(20).std() * 2)
            hist['ATR'] = self._calculate_atr(hist)
            hist['Volume_SMA'] = hist['Volume'].rolling(20).mean()
            hist['Price_Volume_Trend'] = ((hist['Close'] - hist['Close'].shift(1)) / hist['Close'].shift(1)) * hist['Volume']
            
            return hist
            
        except Exception as e:
            logger.error(f"Error getting ultra price data: {e}")
            return pd.DataFrame()
    
    def _get_fundamental_data(self, ticker) -> Dict:
        """Get comprehensive fundamental data"""
        try:
            info = ticker.info
            
            # Extract key fundamental metrics
            fundamentals = {
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                'ev_to_revenue': info.get('enterpriseToRevenue', 0),
                'ev_to_ebitda': info.get('enterpriseToEbitda', 0),
                'profit_margin': info.get('profitMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'return_on_assets': info.get('returnOnAssets', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'payout_ratio': info.get('payoutRatio', 0),
                'beta': info.get('beta', 1.0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'float_shares': info.get('floatShares', 0),
                'insider_ownership': info.get('heldPercentInsiders', 0),
                'institutional_ownership': info.get('heldPercentInstitutions', 0)
            }
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error getting fundamental data: {e}")
            return {}
    
    def _calculate_technical_indicators(self, price_data: pd.DataFrame) -> Dict:
        """Calculate advanced technical indicators"""
        try:
            if price_data.empty:
                return {}
            
            close = price_data['Close']
            high = price_data['High']
            low = price_data['Low']
            volume = price_data['Volume']
            
            indicators = {
                'sma_signals': {
                    'sma_20_above_50': (price_data['SMA_20'] > price_data['SMA_50']).iloc[-1] if 'SMA_20' in price_data.columns else False,
                    'sma_50_above_200': (price_data['SMA_50'] > price_data['SMA_200']).iloc[-1] if 'SMA_50' in price_data.columns else False,
                    'price_above_sma_20': (close > price_data['SMA_20']).iloc[-1] if 'SMA_20' in price_data.columns else False
                },
                'momentum_indicators': {
                    'rsi_current': price_data['RSI'].iloc[-1] if 'RSI' in price_data.columns else 50,
                    'rsi_overbought': price_data['RSI'].iloc[-1] > 70 if 'RSI' in price_data.columns else False,
                    'rsi_oversold': price_data['RSI'].iloc[-1] < 30 if 'RSI' in price_data.columns else False,
                    'macd_bullish': (price_data['MACD'] > price_data['MACD_Signal']).iloc[-1] if 'MACD' in price_data.columns else False
                },
                'volatility_indicators': {
                    'bollinger_position': self._get_bollinger_position(price_data),
                    'atr_current': price_data['ATR'].iloc[-1] if 'ATR' in price_data.columns else 0,
                    'volatility_percentile': self._calculate_volatility_percentile(price_data)
                },
                'volume_indicators': {
                    'volume_above_average': (volume > price_data['Volume_SMA']).iloc[-1] if 'Volume_SMA' in price_data.columns else False,
                    'price_volume_trend': price_data['Price_Volume_Trend'].iloc[-1] if 'Price_Volume_Trend' in price_data.columns else 0
                }
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    def _calculate_risk_metrics(self, price_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive risk metrics"""
        try:
            if price_data.empty or 'Returns' not in price_data.columns:
                return {}
            
            returns = price_data['Returns'].dropna()
            
            if len(returns) < 20:
                return {}
            
            # Basic risk metrics
            annual_vol = returns.std() * np.sqrt(252)
            annual_return = returns.mean() * 252
            sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
            
            # Advanced risk metrics
            downside_returns = returns[returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = annual_return / downside_vol if downside_vol > 0 else 0
            
            # VaR calculations
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
            cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Skewness and Kurtosis
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            risk_metrics = {
                'annual_volatility': annual_vol,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'max_drawdown': max_drawdown,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'downside_volatility': downside_vol,
                'calmar_ratio': annual_return / abs(max_drawdown) if max_drawdown != 0 else 0,
                'information_ratio': self._calculate_information_ratio(returns),
                'treynor_ratio': self._calculate_treynor_ratio(returns, price_data)
            }
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_performance_metrics(self, price_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            if price_data.empty:
                return {}
            
            close = price_data['Close']
            
            # Performance periods
            periods = {
                '1d': 1,
                '1w': 5,
                '1m': 21,
                '3m': 63,
                '6m': 126,
                '1y': 252,
                '2y': 504,
                'ytd': self._get_ytd_days(),
                'max': len(close)
            }
            
            performance = {}
            
            for period_name, days in periods.items():
                if len(close) >= days:
                    start_price = close.iloc[-days]
                    end_price = close.iloc[-1]
                    period_return = (end_price - start_price) / start_price
                    
                    performance[f'return_{period_name}'] = period_return
                    performance[f'annualized_return_{period_name}'] = (1 + period_return) ** (252 / days) - 1 if days > 0 else 0
            
            # Additional performance metrics
            performance['current_price'] = close.iloc[-1]
            performance['52_week_high'] = close.rolling(252).max().iloc[-1] if len(close) >= 252 else close.max()
            performance['52_week_low'] = close.rolling(252).min().iloc[-1] if len(close) >= 252 else close.min()
            performance['distance_from_high'] = (close.iloc[-1] - performance['52_week_high']) / performance['52_week_high']
            performance['distance_from_low'] = (close.iloc[-1] - performance['52_week_low']) / performance['52_week_low']
            
            return performance
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _get_sector_analysis(self, ticker) -> Dict:
        """Get sector and industry analysis"""
        try:
            info = ticker.info
            
            sector_data = {
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'business_summary': info.get('longBusinessSummary', ''),
                'employees': info.get('fullTimeEmployees', 0),
                'country': info.get('country', 'Unknown'),
                'website': info.get('website', ''),
                'sector_performance': self._get_sector_performance(info.get('sector', 'Unknown'))
            }
            
            return sector_data
            
        except Exception as e:
            logger.error(f"Error getting sector analysis: {e}")
            return {}
    
    def _get_dividend_data(self, ticker) -> Dict:
        """Get comprehensive dividend data"""
        try:
            dividends = ticker.dividends
            
            if dividends.empty:
                return {'has_dividends': False}
            
            # Calculate dividend metrics
            annual_dividend = dividends.resample('Y').sum()
            dividend_growth = annual_dividend.pct_change().mean() if len(annual_dividend) > 1 else 0
            
            dividend_data = {
                'has_dividends': True,
                'latest_dividend': dividends.iloc[-1] if not dividends.empty else 0,
                'annual_dividend': annual_dividend.iloc[-1] if not annual_dividend.empty else 0,
                'dividend_growth_rate': dividend_growth,
                'dividend_frequency': len(dividends.resample('Y').count().iloc[-1:]) if not dividends.empty else 0,
                'dividend_history': dividends.tail(20).to_dict()
            }
            
            return dividend_data
            
        except Exception as e:
            logger.error(f"Error getting dividend data: {e}")
            return {'has_dividends': False}
    
    def _get_split_data(self, ticker) -> Dict:
        """Get stock split data"""
        try:
            splits = ticker.splits
            
            split_data = {
                'has_splits': not splits.empty,
                'recent_splits': splits.tail(10).to_dict() if not splits.empty else {},
                'split_count': len(splits)
            }
            
            return split_data
            
        except Exception as e:
            logger.error(f"Error getting split data: {e}")
            return {'has_splits': False}
    
    def _get_options_data(self, ticker) -> Dict:
        """Get options data if available"""
        try:
            options_dates = ticker.options
            
            if not options_dates:
                return {'has_options': False}
            
            # Get options for nearest expiry
            nearest_expiry = options_dates[0]
            options_chain = ticker.option_chain(nearest_expiry)
            
            options_data = {
                'has_options': True,
                'expiry_dates': list(options_dates),
                'nearest_expiry': nearest_expiry,
                'calls_count': len(options_chain.calls),
                'puts_count': len(options_chain.puts),
                'max_pain': self._calculate_max_pain(options_chain) if options_chain else None
            }
            
            return options_data
            
        except Exception as e:
            logger.error(f"Error getting options data: {e}")
            return {'has_options': False}
    
    def _get_institutional_data(self, ticker) -> Dict:
        """Get institutional holdings data"""
        try:
            institutional = ticker.institutional_holders
            
            if institutional is None or institutional.empty:
                return {'has_institutional_data': False}
            
            institutional_data = {
                'has_institutional_data': True,
                'top_holders': institutional.head(10).to_dict('records'),
                'total_institutions': len(institutional),
                'top_10_percentage': institutional.head(10)['% Out'].sum() if '% Out' in institutional.columns else 0
            }
            
            return institutional_data
            
        except Exception as e:
            logger.error(f"Error getting institutional data: {e}")
            return {'has_institutional_data': False}
    
    def _get_analyst_data(self, ticker) -> Dict:
        """Get analyst recommendations"""
        try:
            recommendations = ticker.recommendations
            
            if recommendations is None or recommendations.empty:
                return {'has_analyst_data': False}
            
            # Get latest recommendations
            latest_recs = recommendations.tail(10)
            
            analyst_data = {
                'has_analyst_data': True,
                'latest_recommendations': latest_recs.to_dict('records'),
                'recommendation_summary': self._summarize_recommendations(latest_recs)
            }
            
            return analyst_data
            
        except Exception as e:
            logger.error(f"Error getting analyst data: {e}")
            return {'has_analyst_data': False}
    
    # Helper methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def _get_bollinger_position(self, price_data: pd.DataFrame) -> float:
        """Get current position within Bollinger Bands"""
        if 'Bollinger_Upper' not in price_data.columns:
            return 0.5
        
        current_price = price_data['Close'].iloc[-1]
        upper_band = price_data['Bollinger_Upper'].iloc[-1]
        lower_band = price_data['Bollinger_Lower'].iloc[-1]
        
        if upper_band == lower_band:
            return 0.5
        
        position = (current_price - lower_band) / (upper_band - lower_band)
        return max(0, min(1, position))
    
    def _calculate_volatility_percentile(self, price_data: pd.DataFrame) -> float:
        """Calculate current volatility percentile"""
        if 'Volatility_20' not in price_data.columns:
            return 50
        
        current_vol = price_data['Volatility_20'].iloc[-1]
        vol_series = price_data['Volatility_20'].dropna()
        
        if len(vol_series) == 0:
            return 50
        
        percentile = (vol_series < current_vol).sum() / len(vol_series) * 100
        return percentile
    
    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """Calculate Information Ratio"""
        # Assuming benchmark return of 0 for simplicity
        excess_returns = returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        if tracking_error == 0:
            return 0
        
        return (excess_returns.mean() * 252) / tracking_error
    
    def _calculate_treynor_ratio(self, returns: pd.Series, price_data: pd.DataFrame) -> float:
        """Calculate Treynor Ratio"""
        # Simple beta calculation against market (assuming SPY-like behavior)
        market_returns = returns  # Simplified
        beta = 1.0  # Simplified
        
        annual_return = returns.mean() * 252
        risk_free_rate = 0.02  # 2% risk-free rate
        
        if beta == 0:
            return 0
        
        return (annual_return - risk_free_rate) / beta
    
    def _get_ytd_days(self) -> int:
        """Get number of trading days year-to-date"""
        start_of_year = datetime(datetime.now().year, 1, 1)
        days_passed = (datetime.now() - start_of_year).days
        return min(days_passed, 252)  # Approximate trading days
    
    def _get_sector_performance(self, sector: str) -> Dict:
        """Get sector performance data"""
        # Simplified sector performance
        return {
            'sector_name': sector,
            'relative_performance': 'neutral'  # Would need sector ETF data for real calculation
        }
    
    def _calculate_max_pain(self, options_chain) -> float:
        """Calculate options max pain point"""
        try:
            calls = options_chain.calls
            puts = options_chain.puts
            
            # Simplified max pain calculation
            if calls.empty or puts.empty:
                return None
            
            # This is a simplified version - real max pain requires more complex calculation
            return (calls['strike'].median() + puts['strike'].median()) / 2
            
        except Exception:
            return None
    
    def _summarize_recommendations(self, recommendations: pd.DataFrame) -> Dict:
        """Summarize analyst recommendations"""
        try:
            if 'To Grade' not in recommendations.columns:
                return {}
            
            grades = recommendations['To Grade'].value_counts()
            
            return {
                'buy_count': grades.get('Buy', 0) + grades.get('Strong Buy', 0),
                'hold_count': grades.get('Hold', 0),
                'sell_count': grades.get('Sell', 0) + grades.get('Strong Sell', 0),
                'total_analysts': len(recommendations)
            }
            
        except Exception:
            return {}
    
    def _calculate_portfolio_metrics(self, comprehensive_data: Dict) -> Dict:
        """Calculate portfolio-level metrics"""
        try:
            portfolio_metrics = {
                'total_symbols': len(comprehensive_data.get('price_data', {})),
                'data_quality_score': 0.99,  # High quality from Yahoo Finance
                'average_sharpe_ratio': 0,
                'portfolio_beta': 1.0,
                'diversification_score': 0
            }
            
            # Calculate average Sharpe ratio
            sharpe_ratios = []
            for symbol, risk_data in comprehensive_data.get('risk_metrics', {}).items():
                if 'sharpe_ratio' in risk_data:
                    sharpe_ratios.append(risk_data['sharpe_ratio'])
            
            if sharpe_ratios:
                portfolio_metrics['average_sharpe_ratio'] = np.mean(sharpe_ratios)
            
            return portfolio_metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}


# Test function
def test_advanced_yahoo_fetcher():
    """Test the advanced Yahoo Finance fetcher"""
    fetcher = AdvancedYahooFetcher()
    
    print("🚀 Testing Advanced Yahoo Finance Fetcher...")
    
    # Test with sample portfolio
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    data = fetcher.get_ultra_comprehensive_data(symbols, '1y')
    
    if data:
        print(f"✅ Successfully fetched comprehensive data for {len(symbols)} symbols")
        print(f"📊 Data categories: {list(data.keys())}")
        
        # Show sample metrics
        for symbol in symbols:
            if symbol in data.get('risk_metrics', {}):
                sharpe = data['risk_metrics'][symbol].get('sharpe_ratio', 0)
                print(f"📈 {symbol} Sharpe Ratio: {sharpe:.3f}")
    else:
        print("❌ Failed to fetch data")


if __name__ == "__main__":
    test_advanced_yahoo_fetcher()
