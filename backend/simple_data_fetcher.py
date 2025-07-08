"""
Simple Data Fetcher - Reliable fallback data fetcher using yfinance
Guaranteed to work with basic functionality
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class SimpleDataFetcher:
    """
    Simple, reliable data fetcher using yfinance
    Designed to always work as a fallback option
    """
    
    def __init__(self):
        self.period_mapping = {
            '1d': '1d',
            '1w': '5d', 
            '1m': '1mo',
            '3m': '3mo',
            '6m': '6mo',
            '1y': '1y',
            '2y': '2y',
            '5y': '5y',
            '10y': '10y'
        }
    
    def fetch_portfolio_data(self, symbols: List[str], period: str = '1y') -> Optional[pd.DataFrame]:
        """
        Fetch portfolio data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            period: Time period for data
            
        Returns:
            DataFrame with price data for all symbols
        """
        try:
            logger.info(f"Fetching data for symbols: {symbols}")
            
            # Convert period to yfinance format
            yf_period = self.period_mapping.get(period, '1y')
            
            # Fetch data for all symbols at once
            tickers = yf.Tickers(' '.join(symbols))
            
            portfolio_data = {}
            
            for symbol in symbols:
                try:
                    logger.info(f"Fetching data for {symbol}")
                    
                    # Get historical data
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=yf_period)
                    
                    if hist.empty:
                        logger.warning(f"No data returned for {symbol}")
                        continue
                    
                    # Use Close price (most reliable column)
                    if 'Close' in hist.columns:
                        portfolio_data[symbol] = hist['Close']
                    else:
                        logger.warning(f"No Close price data for {symbol}")
                        continue
                    
                    logger.info(f"Successfully fetched {len(hist)} data points for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
                    continue
            
            if not portfolio_data:
                logger.error("No data fetched for any symbols")
                return None
            
            # Combine into DataFrame
            df = pd.DataFrame(portfolio_data)
            
            # Remove rows with any NaN values
            df = df.dropna()
            
            if df.empty:
                logger.error("No valid data after cleaning")
                return None
            
            logger.info(f"Portfolio data ready: {df.shape[0]} rows, {df.shape[1]} symbols")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in fetch_portfolio_data: {e}")
            return None
    
    def fetch_single_stock_data(self, symbol: str, period: str = '1y') -> Optional[dict]:
        """
        Fetch data for a single stock
        
        Args:
            symbol: Stock symbol
            period: Time period
            
        Returns:
            Dictionary with stock data
        """
        try:
            logger.info(f"Fetching single stock data for {symbol}")
            
            yf_period = self.period_mapping.get(period, '1y')
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=yf_period)
            
            if hist.empty:
                logger.warning(f"No historical data for {symbol}")
                return None
            
            # Get basic info
            try:
                info = ticker.info
            except:
                info = {}
            
            # Calculate returns
            if 'Close' in hist.columns:
                returns = hist['Close'].pct_change().dropna()
            else:
                returns = pd.Series(dtype=float)
            
            # Format data
            stock_data = {
                'symbol': symbol,
                'prices': hist.to_dict('index'),
                'returns': returns.to_dict() if not returns.empty else {},
                'summary': {
                    'current_price': float(hist['Close'].iloc[-1]) if 'Close' in hist.columns else 0,
                    'previous_close': float(hist['Close'].iloc[-2]) if len(hist) > 1 and 'Close' in hist.columns else 0,
                    'change': float(hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) if len(hist) > 1 and 'Close' in hist.columns else 0,
                    'change_percent': float(returns.iloc[-1] * 100) if not returns.empty else 0,
                    'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
                    'high_52w': float(hist['High'].max()) if 'High' in hist.columns else 0,
                    'low_52w': float(hist['Low'].min()) if 'Low' in hist.columns else 0,
                    'avg_volume': float(hist['Volume'].mean()) if 'Volume' in hist.columns else 0,
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'dividend_yield': info.get('dividendYield', 0)
                },
                'info': info,
                'last_updated': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully formatted data for {symbol}")
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching single stock data for {symbol}: {e}")
            return None
    
    def test_connection(self) -> bool:
        """
        Test if data fetching is working
        
        Returns:
            True if connection works, False otherwise
        """
        try:
            logger.info("Testing data fetcher connection...")
            
            # Try to fetch a small amount of data for a reliable stock
            ticker = yf.Ticker("AAPL")
            hist = ticker.history(period="5d")
            
            if not hist.empty:
                logger.info("✅ Data fetcher connection test successful")
                return True
            else:
                logger.warning("❌ Data fetcher connection test failed - no data returned")
                return False
                
        except Exception as e:
            logger.error(f"❌ Data fetcher connection test failed: {e}")
            return False
    
    def get_available_symbols(self) -> List[str]:
        """
        Get a list of commonly available symbols for testing
        
        Returns:
            List of stock symbols that should be available
        """
        return [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
            'META', 'NVDA', 'NFLX', 'AMD', 'INTC',
            'SPY', 'QQQ', 'VTI', 'IWM', 'DIA'
        ]
    
    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """
        Validate which symbols are available
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            List of valid symbols
        """
        valid_symbols = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                
                if not hist.empty:
                    valid_symbols.append(symbol)
                    logger.info(f"✅ {symbol} is valid")
                else:
                    logger.warning(f"❌ {symbol} returned no data")
                    
            except Exception as e:
                logger.warning(f"❌ {symbol} validation failed: {e}")
        
        return valid_symbols


# Test function
def test_simple_fetcher():
    """Test the simple data fetcher"""
    print("Testing Simple Data Fetcher...")
    
    fetcher = SimpleDataFetcher()
    
    # Test connection
    if not fetcher.test_connection():
        print("❌ Connection test failed")
        return False
    
    # Test portfolio data fetch
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    data = fetcher.fetch_portfolio_data(symbols, '1m')
    
    if data is not None:
        print(f"✅ Portfolio data fetched successfully: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Sample data:\n{data.head()}")
        return True
    else:
        print("❌ Portfolio data fetch failed")
        return False


if __name__ == "__main__":
    test_simple_fetcher()
