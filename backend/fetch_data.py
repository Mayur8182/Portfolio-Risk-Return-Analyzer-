"""
Portfolio Data Fetcher - Handles market data retrieval from various sources
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import pickle
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class PortfolioDataFetcher:
    """
    Handles fetching and caching of market data for portfolio analysis
    """
    
    def __init__(self, cache_dir: str = "../data/cache"):
        self.cache_dir = cache_dir
        self.cache_duration = timedelta(hours=1)  # Cache data for 1 hour
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, symbol: str, period: str) -> str:
        """Generate cache file path for a symbol and period"""
        return os.path.join(self.cache_dir, f"{symbol}_{period}.pkl")
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cached data is still valid"""
        if not os.path.exists(cache_path):
            return False
        
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - cache_time < self.cache_duration
    
    def _save_to_cache(self, data: pd.DataFrame, cache_path: str):
        """Save data to cache"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _load_from_cache(self, cache_path: str) -> Optional[pd.DataFrame]:
        """Load data from cache"""
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def fetch_single_stock_data(self, symbol: str, period: str = "1y") -> Dict:
        """
        Fetch data for a single stock symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        
        Returns:
            Dictionary containing stock data and metadata
        """
        cache_path = self._get_cache_path(symbol, period)
        
        # Try to load from cache first
        if self._is_cache_valid(cache_path):
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                logger.info(f"Loaded {symbol} data from cache")
                return self._format_stock_data(cached_data, symbol)
        
        try:
            # Fetch fresh data from Yahoo Finance
            logger.info(f"Fetching fresh data for {symbol}")
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            hist_data = ticker.history(period=period)
            
            if hist_data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Get additional info
            info = ticker.info
            
            # Save to cache
            self._save_to_cache(hist_data, cache_path)
            
            return self._format_stock_data(hist_data, symbol, info)
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def fetch_portfolio_data(self, symbols: List[str], period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Fetch data for multiple stocks in a portfolio
        
        Args:
            symbols: List of stock symbols
            period: Time period for historical data
        
        Returns:
            DataFrame with adjusted close prices for all symbols
        """
        try:
            logger.info(f"Fetching portfolio data for: {symbols}")
            
            # Fetch data for all symbols
            portfolio_data = {}
            
            for symbol in symbols:
                stock_data = self.fetch_single_stock_data(symbol, period)
                if stock_data is None:
                    logger.error(f"Failed to fetch data for {symbol}")
                    return None
                
                # Try different column names for price data
                prices_dict = stock_data['prices']
                if 'Adj Close' in prices_dict:
                    portfolio_data[symbol] = prices_dict['Adj Close']
                elif 'Close' in prices_dict:
                    portfolio_data[symbol] = prices_dict['Close']
                else:
                    # If neither exists, try to get the first price column
                    price_columns = [col for col in prices_dict.keys() if isinstance(col, str) and ('close' in col.lower() or 'price' in col.lower())]
                    if price_columns:
                        portfolio_data[symbol] = prices_dict[price_columns[0]]
                    else:
                        logger.error(f"No price data found for {symbol}")
                        return None
            
            # Combine into a single DataFrame
            df = pd.DataFrame(portfolio_data)
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            if df.empty:
                raise ValueError("No valid data found for portfolio")
            
            logger.info(f"Successfully fetched data for {len(symbols)} stocks, {len(df)} data points")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching portfolio data: {e}")
            return None
    
    def _format_stock_data(self, hist_data: pd.DataFrame, symbol: str, info: Dict = None) -> Dict:
        """
        Format stock data into a standardized structure
        """
        # Calculate additional metrics - handle different column names
        price_column = None
        if 'Adj Close' in hist_data.columns:
            price_column = 'Adj Close'
        elif 'Close' in hist_data.columns:
            price_column = 'Close'
        else:
            # Find any column with 'close' or 'price' in the name
            price_columns = [col for col in hist_data.columns if isinstance(col, str) and ('close' in col.lower() or 'price' in col.lower())]
            if price_columns:
                price_column = price_columns[0]
            else:
                logger.error(f"No price column found in data for {symbol}")
                price_column = hist_data.columns[0]  # Use first column as fallback

        returns = hist_data[price_column].pct_change().dropna()
        
        formatted_data = {
            'symbol': symbol,
            'prices': hist_data.to_dict('index'),
            'returns': returns.to_dict(),
            'summary': {
                'current_price': float(hist_data[price_column].iloc[-1]),
                'previous_close': float(hist_data[price_column].iloc[-2]) if len(hist_data) > 1 else None,
                'change': float(hist_data[price_column].iloc[-1] - hist_data[price_column].iloc[-2]) if len(hist_data) > 1 else 0,
                'change_percent': float(returns.iloc[-1] * 100) if len(returns) > 0 else 0,
                'volume': int(hist_data['Volume'].iloc[-1]) if 'Volume' in hist_data.columns else 0,
                'high_52w': float(hist_data['High'].max()) if 'High' in hist_data.columns else float(hist_data[price_column].max()),
                'low_52w': float(hist_data['Low'].min()) if 'Low' in hist_data.columns else float(hist_data[price_column].min()),
                'avg_volume': float(hist_data['Volume'].mean()) if 'Volume' in hist_data.columns else 0,
                'volatility': float(returns.std() * np.sqrt(252)),  # Annualized volatility
            },
            'metadata': {
                'start_date': hist_data.index[0].strftime('%Y-%m-%d'),
                'end_date': hist_data.index[-1].strftime('%Y-%m-%d'),
                'data_points': len(hist_data),
                'last_updated': datetime.now().isoformat()
            }
        }
        
        # Add company info if available
        if info:
            formatted_data['company_info'] = {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'dividend_yield': info.get('dividendYield', None),
                'beta': info.get('beta', None)
            }
        
        return formatted_data
    
    def get_market_indices(self, period: str = "1y") -> Dict:
        """
        Fetch major market indices for benchmarking
        """
        indices = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ 100',
            'DIA': 'Dow Jones',
            'VTI': 'Total Stock Market',
            'IWM': 'Russell 2000'
        }
        
        market_data = {}
        
        for symbol, name in indices.items():
            data = self.fetch_single_stock_data(symbol, period)
            if data:
                market_data[symbol] = {
                    'name': name,
                    'data': data
                }
        
        return market_data
    
    def validate_symbols(self, symbols: List[str]) -> Dict[str, bool]:
        """
        Validate if stock symbols exist and have data
        """
        validation_results = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Check if we got valid info
                validation_results[symbol] = bool(info.get('regularMarketPrice') or info.get('previousClose'))
                
            except Exception:
                validation_results[symbol] = False
        
        return validation_results
