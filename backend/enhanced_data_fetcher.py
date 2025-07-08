"""
Enhanced Data Fetcher - World-class financial data integration with Finnhub and Twelve Data APIs
Provides institutional-grade accuracy with real-time data and comprehensive deduplication
"""

import finnhub
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import asyncio
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
from typing import Dict, List, Optional, Tuple
import hashlib
import time
from dataclasses import dataclass
from pymongo import MongoClient
import os

logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
    """Standardized market data structure"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    adjusted_close: float
    source: str
    data_hash: str

class EnhancedDataFetcher:
    """
    World-class financial data fetcher with multiple premium data sources
    Features:
    - Real-time data from Finnhub and Twelve Data
    - Comprehensive deduplication
    - MongoDB caching and storage
    - Data quality validation
    - Institutional-grade accuracy
    """
    
    def __init__(self):
        # API Configuration
        self.finnhub_api_key = "d1hodf1r01qsvr2be4r0d1hodf1r01qsvr2be4rg"
        self.finnhub_secret = "d1hodf1r01qsvr2be4sg"
        self.twelvedata_api_key = "b57491954ba64f1d9c851fb4c157bf51"
        
        # Initialize API clients
        self.finnhub_client = finnhub.Client(api_key=self.finnhub_api_key)
        
        # MongoDB connection
        self.mongo_client = MongoClient('mongodb://localhost:27017/')
        self.db = self.mongo_client['portfolio_analytics']
        self.market_data_collection = self.db['market_data']
        self.company_profiles_collection = self.db['company_profiles']
        self.real_time_quotes_collection = self.db['real_time_quotes']
        
        # Create indexes for performance
        self._create_indexes()
        
        # Data quality thresholds
        self.min_volume_threshold = 1000
        self.max_price_change_threshold = 0.5  # 50% daily change filter
        
        # Rate limiting
        self.last_finnhub_call = 0
        self.last_twelvedata_call = 0
        self.finnhub_rate_limit = 1.0  # 1 second between calls
        self.twelvedata_rate_limit = 0.1  # 100ms between calls
    
    def _create_indexes(self):
        """Create MongoDB indexes for optimal performance"""
        try:
            # Market data indexes
            self.market_data_collection.create_index([("symbol", 1), ("timestamp", -1)])
            self.market_data_collection.create_index([("data_hash", 1)], unique=True)
            self.market_data_collection.create_index([("source", 1)])
            
            # Company profiles indexes
            self.company_profiles_collection.create_index([("symbol", 1)], unique=True)
            
            # Real-time quotes indexes
            self.real_time_quotes_collection.create_index([("symbol", 1)])
            self.real_time_quotes_collection.create_index([("timestamp", -1)])
            
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating MongoDB indexes: {e}")
    
    def _generate_data_hash(self, symbol: str, timestamp: datetime, 
                          open_price: float, high: float, low: float, 
                          close: float, volume: int) -> str:
        """Generate unique hash for data deduplication"""
        data_string = f"{symbol}_{timestamp.isoformat()}_{open_price}_{high}_{low}_{close}_{volume}"
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def _rate_limit_finnhub(self):
        """Implement rate limiting for Finnhub API"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_finnhub_call
        if time_since_last_call < self.finnhub_rate_limit:
            time.sleep(self.finnhub_rate_limit - time_since_last_call)
        self.last_finnhub_call = time.time()
    
    def _rate_limit_twelvedata(self):
        """Implement rate limiting for Twelve Data API"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_twelvedata_call
        if time_since_last_call < self.twelvedata_rate_limit:
            time.sleep(self.twelvedata_rate_limit - time_since_last_call)
        self.last_twelvedata_call = time.time()
    
    def _validate_data_quality(self, data_point: MarketDataPoint) -> bool:
        """Validate data quality with institutional standards"""
        try:
            # Check for valid prices
            if any(price <= 0 for price in [data_point.open_price, data_point.high_price, 
                                          data_point.low_price, data_point.close_price]):
                return False
            
            # Check price relationships
            if not (data_point.low_price <= data_point.open_price <= data_point.high_price and
                   data_point.low_price <= data_point.close_price <= data_point.high_price):
                return False
            
            # Check volume threshold
            if data_point.volume < self.min_volume_threshold:
                return False
            
            # Check for extreme price movements (potential data errors)
            if data_point.open_price > 0:
                daily_change = abs(data_point.close_price - data_point.open_price) / data_point.open_price
                if daily_change > self.max_price_change_threshold:
                    logger.warning(f"Extreme price movement detected for {data_point.symbol}: {daily_change:.2%}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data quality: {e}")
            return False
    
    def get_real_time_quote_finnhub(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote from Finnhub with highest accuracy"""
        try:
            self._rate_limit_finnhub()
            
            # Get real-time quote
            quote = self.finnhub_client.quote(symbol)
            
            if not quote or 'c' not in quote:
                return None
            
            # Get company profile for additional context
            profile = self.finnhub_client.company_profile2(symbol=symbol)
            
            real_time_data = {
                'symbol': symbol,
                'current_price': quote['c'],  # Current price
                'change': quote['d'],         # Change
                'percent_change': quote['dp'], # Percent change
                'high_price': quote['h'],     # High price of the day
                'low_price': quote['l'],      # Low price of the day
                'open_price': quote['o'],     # Open price of the day
                'previous_close': quote['pc'], # Previous close price
                'timestamp': datetime.now(),
                'source': 'finnhub',
                'company_name': profile.get('name', '') if profile else '',
                'market_cap': profile.get('marketCapitalization', 0) if profile else 0,
                'industry': profile.get('finnhubIndustry', '') if profile else '',
                'exchange': profile.get('exchange', '') if profile else ''
            }
            
            # Store in MongoDB
            self._store_real_time_quote(real_time_data)
            
            return real_time_data
            
        except Exception as e:
            logger.error(f"Error fetching real-time quote from Finnhub for {symbol}: {e}")
            return None
    
    def get_historical_data_twelvedata(self, symbol: str, interval: str = "1day", 
                                     outputsize: int = 5000) -> Optional[pd.DataFrame]:
        """Get historical data from Twelve Data with premium accuracy"""
        try:
            self._rate_limit_twelvedata()
            
            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': symbol,
                'interval': interval,
                'outputsize': outputsize,
                'apikey': self.twelvedata_api_key,
                'format': 'JSON'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'values' not in data:
                logger.error(f"No data returned for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            
            # Convert to numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Store in MongoDB with deduplication
            self._store_historical_data(symbol, df, 'twelvedata')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data from Twelve Data for {symbol}: {e}")
            return None
    
    def get_comprehensive_company_data_finnhub(self, symbol: str) -> Optional[Dict]:
        """Get comprehensive company data from Finnhub"""
        try:
            self._rate_limit_finnhub()
            
            # Company profile
            profile = self.finnhub_client.company_profile2(symbol=symbol)
            
            # Financial metrics
            metrics = self.finnhub_client.company_basic_financials(symbol, 'all')
            
            # Recommendation trends
            recommendations = self.finnhub_client.recommendation_trends(symbol)
            
            # Price target
            price_target = self.finnhub_client.price_target(symbol)
            
            comprehensive_data = {
                'symbol': symbol,
                'profile': profile,
                'metrics': metrics,
                'recommendations': recommendations,
                'price_target': price_target,
                'last_updated': datetime.now(),
                'source': 'finnhub'
            }
            
            # Store in MongoDB
            self._store_company_profile(comprehensive_data)
            
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Error fetching comprehensive data from Finnhub for {symbol}: {e}")
            return None
    
    def _store_real_time_quote(self, quote_data: Dict):
        """Store real-time quote in MongoDB"""
        try:
            # Add unique identifier
            quote_data['_id'] = f"{quote_data['symbol']}_{quote_data['timestamp'].isoformat()}"
            
            # Upsert to avoid duplicates
            self.real_time_quotes_collection.replace_one(
                {'_id': quote_data['_id']},
                quote_data,
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Error storing real-time quote: {e}")
    
    def _store_historical_data(self, symbol: str, df: pd.DataFrame, source: str):
        """Store historical data in MongoDB with deduplication"""
        try:
            documents = []
            
            for timestamp, row in df.iterrows():
                # Generate hash for deduplication
                data_hash = self._generate_data_hash(
                    symbol, timestamp, row['open'], row['high'], 
                    row['low'], row['close'], int(row['volume'])
                )
                
                data_point = MarketDataPoint(
                    symbol=symbol,
                    timestamp=timestamp,
                    open_price=float(row['open']),
                    high_price=float(row['high']),
                    low_price=float(row['low']),
                    close_price=float(row['close']),
                    volume=int(row['volume']),
                    adjusted_close=float(row['close']),  # Twelve Data doesn't separate adjusted close
                    source=source,
                    data_hash=data_hash
                )
                
                # Validate data quality
                if self._validate_data_quality(data_point):
                    doc = {
                        'symbol': data_point.symbol,
                        'timestamp': data_point.timestamp,
                        'open': data_point.open_price,
                        'high': data_point.high_price,
                        'low': data_point.low_price,
                        'close': data_point.close_price,
                        'volume': data_point.volume,
                        'adjusted_close': data_point.adjusted_close,
                        'source': data_point.source,
                        'data_hash': data_point.data_hash,
                        'created_at': datetime.now()
                    }
                    documents.append(doc)
            
            # Bulk insert with ignore duplicates
            if documents:
                try:
                    self.market_data_collection.insert_many(documents, ordered=False)
                    logger.info(f"Stored {len(documents)} data points for {symbol}")
                except Exception as e:
                    # Handle duplicate key errors gracefully
                    if "duplicate key error" in str(e).lower():
                        logger.info(f"Some duplicate data points skipped for {symbol}")
                    else:
                        raise e
                        
        except Exception as e:
            logger.error(f"Error storing historical data: {e}")
    
    def _store_company_profile(self, company_data: Dict):
        """Store company profile in MongoDB"""
        try:
            # Upsert company profile
            self.company_profiles_collection.replace_one(
                {'symbol': company_data['symbol']},
                company_data,
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Error storing company profile: {e}")
    
    def get_portfolio_data_enhanced(self, symbols: List[str], period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Get enhanced portfolio data with world-class accuracy
        Combines multiple data sources for maximum reliability
        """
        try:
            all_data = {}
            
            for symbol in symbols:
                logger.info(f"Fetching enhanced data for {symbol}")
                
                # Try Twelve Data first for historical data
                df = self.get_historical_data_twelvedata(symbol, interval="1day")
                
                if df is not None and not df.empty:
                    # Get real-time quote for latest price
                    real_time = self.get_real_time_quote_finnhub(symbol)
                    
                    # Get comprehensive company data
                    company_data = self.get_comprehensive_company_data_finnhub(symbol)
                    
                    # Use adjusted close or close price
                    all_data[symbol] = df['close']
                    
                    logger.info(f"Successfully fetched {len(df)} data points for {symbol}")
                else:
                    logger.warning(f"No data available for {symbol}")
                    continue
            
            if not all_data:
                logger.error("No data fetched for any symbols")
                return None
            
            # Combine all data into single DataFrame
            portfolio_df = pd.DataFrame(all_data)
            
            # Remove any rows with NaN values
            portfolio_df = portfolio_df.dropna()
            
            if portfolio_df.empty:
                logger.error("No valid data after cleaning")
                return None
            
            logger.info(f"Enhanced portfolio data ready: {len(portfolio_df)} data points for {len(symbols)} symbols")
            return portfolio_df
            
        except Exception as e:
            logger.error(f"Error fetching enhanced portfolio data: {e}")
            return None
    
    def get_cached_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Retrieve cached data from MongoDB"""
        try:
            query = {
                'symbol': symbol,
                'timestamp': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
            
            cursor = self.market_data_collection.find(query).sort('timestamp', 1)
            data = list(cursor)
            
            if not data:
                return None
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume', 'adjusted_close']]
            
        except Exception as e:
            logger.error(f"Error retrieving cached data: {e}")
            return None
