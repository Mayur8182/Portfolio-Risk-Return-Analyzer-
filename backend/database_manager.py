"""
MongoDB Database Manager - World-class data storage and management
Provides institutional-grade data persistence, caching, and analytics storage
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, ConnectionFailure
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import gridfs
from bson import ObjectId
import json

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Advanced MongoDB manager for portfolio analytics
    Features:
    - High-performance indexing
    - Data deduplication
    - Automated archiving
    - Real-time analytics storage
    - GridFS for large datasets
    """
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/"):
        try:
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
            
            self.db = self.client['portfolio_analytics_pro']
            
            # Initialize collections
            self._initialize_collections()
            
            # Initialize GridFS for large files
            self.fs = gridfs.GridFS(self.db)
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _initialize_collections(self):
        """Initialize all collections with proper schemas and indexes"""
        
        # Market Data Collection
        self.market_data = self.db['market_data']
        self._create_market_data_indexes()
        
        # Real-time Quotes Collection
        self.real_time_quotes = self.db['real_time_quotes']
        self._create_real_time_indexes()
        
        # Company Profiles Collection
        self.company_profiles = self.db['company_profiles']
        self._create_company_indexes()
        
        # Portfolio Analysis Results
        self.portfolio_analyses = self.db['portfolio_analyses']
        self._create_analysis_indexes()
        
        # Risk Metrics Historical
        self.risk_metrics_history = self.db['risk_metrics_history']
        self._create_risk_metrics_indexes()
        
        # Optimization Results
        self.optimization_results = self.db['optimization_results']
        self._create_optimization_indexes()
        
        # Sentiment Analysis Data
        self.sentiment_data = self.db['sentiment_data']
        self._create_sentiment_indexes()
        
        # Data Quality Logs
        self.data_quality_logs = self.db['data_quality_logs']
        self._create_quality_indexes()
        
        # User Sessions and Analytics
        self.user_sessions = self.db['user_sessions']
        self._create_session_indexes()
        
        logger.info("All collections initialized successfully")
    
    def _create_market_data_indexes(self):
        """Create optimized indexes for market data"""
        try:
            # Compound index for symbol and timestamp (most common query)
            self.market_data.create_index([
                ("symbol", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="symbol_timestamp_idx")
            
            # Unique index for deduplication
            self.market_data.create_index([
                ("data_hash", ASCENDING)
            ], unique=True, name="data_hash_unique_idx")
            
            # Source index for data lineage
            self.market_data.create_index([
                ("source", ASCENDING)
            ], name="source_idx")
            
            # Volume index for filtering
            self.market_data.create_index([
                ("volume", DESCENDING)
            ], name="volume_idx")
            
            # Date range index for time-series queries
            self.market_data.create_index([
                ("timestamp", DESCENDING)
            ], name="timestamp_idx")
            
            # Compound index for price analysis
            self.market_data.create_index([
                ("symbol", ASCENDING),
                ("close", DESCENDING),
                ("timestamp", DESCENDING)
            ], name="symbol_price_time_idx")
            
            logger.info("Market data indexes created")
            
        except Exception as e:
            logger.error(f"Error creating market data indexes: {e}")
    
    def _create_real_time_indexes(self):
        """Create indexes for real-time quotes"""
        try:
            self.real_time_quotes.create_index([
                ("symbol", ASCENDING)
            ], name="rt_symbol_idx")
            
            self.real_time_quotes.create_index([
                ("timestamp", DESCENDING)
            ], name="rt_timestamp_idx")
            
            # TTL index for automatic cleanup (keep real-time data for 7 days)
            self.real_time_quotes.create_index([
                ("timestamp", ASCENDING)
            ], expireAfterSeconds=604800, name="rt_ttl_idx")  # 7 days
            
            logger.info("Real-time quotes indexes created")
            
        except Exception as e:
            logger.error(f"Error creating real-time indexes: {e}")
    
    def _create_company_indexes(self):
        """Create indexes for company profiles"""
        try:
            self.company_profiles.create_index([
                ("symbol", ASCENDING)
            ], unique=True, name="company_symbol_unique_idx")
            
            self.company_profiles.create_index([
                ("profile.marketCapitalization", DESCENDING)
            ], name="market_cap_idx")
            
            self.company_profiles.create_index([
                ("profile.finnhubIndustry", ASCENDING)
            ], name="industry_idx")
            
            logger.info("Company profiles indexes created")
            
        except Exception as e:
            logger.error(f"Error creating company indexes: {e}")
    
    def _create_analysis_indexes(self):
        """Create indexes for portfolio analyses"""
        try:
            self.portfolio_analyses.create_index([
                ("portfolio_hash", ASCENDING)
            ], name="portfolio_hash_idx")
            
            self.portfolio_analyses.create_index([
                ("created_at", DESCENDING)
            ], name="analysis_created_idx")
            
            self.portfolio_analyses.create_index([
                ("symbols", ASCENDING)
            ], name="symbols_idx")
            
            # TTL index for analysis cleanup (keep for 30 days)
            self.portfolio_analyses.create_index([
                ("created_at", ASCENDING)
            ], expireAfterSeconds=2592000, name="analysis_ttl_idx")  # 30 days
            
            logger.info("Portfolio analysis indexes created")
            
        except Exception as e:
            logger.error(f"Error creating analysis indexes: {e}")
    
    def _create_risk_metrics_indexes(self):
        """Create indexes for risk metrics history"""
        try:
            self.risk_metrics_history.create_index([
                ("portfolio_id", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="portfolio_risk_time_idx")
            
            self.risk_metrics_history.create_index([
                ("sharpe_ratio", DESCENDING)
            ], name="sharpe_ratio_idx")
            
            self.risk_metrics_history.create_index([
                ("volatility", ASCENDING)
            ], name="volatility_idx")
            
            logger.info("Risk metrics indexes created")
            
        except Exception as e:
            logger.error(f"Error creating risk metrics indexes: {e}")
    
    def _create_optimization_indexes(self):
        """Create indexes for optimization results"""
        try:
            self.optimization_results.create_index([
                ("portfolio_hash", ASCENDING),
                ("objective", ASCENDING)
            ], name="portfolio_objective_idx")
            
            self.optimization_results.create_index([
                ("created_at", DESCENDING)
            ], name="optimization_created_idx")
            
            logger.info("Optimization results indexes created")
            
        except Exception as e:
            logger.error(f"Error creating optimization indexes: {e}")
    
    def _create_sentiment_indexes(self):
        """Create indexes for sentiment data"""
        try:
            self.sentiment_data.create_index([
                ("symbol", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="sentiment_symbol_time_idx")
            
            self.sentiment_data.create_index([
                ("overall_score", DESCENDING)
            ], name="sentiment_score_idx")
            
            # TTL index for sentiment cleanup (keep for 14 days)
            self.sentiment_data.create_index([
                ("timestamp", ASCENDING)
            ], expireAfterSeconds=1209600, name="sentiment_ttl_idx")  # 14 days
            
            logger.info("Sentiment data indexes created")
            
        except Exception as e:
            logger.error(f"Error creating sentiment indexes: {e}")
    
    def _create_quality_indexes(self):
        """Create indexes for data quality logs"""
        try:
            self.data_quality_logs.create_index([
                ("symbol", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="quality_symbol_time_idx")
            
            self.data_quality_logs.create_index([
                ("quality_score", DESCENDING)
            ], name="quality_score_idx")
            
            logger.info("Data quality indexes created")
            
        except Exception as e:
            logger.error(f"Error creating quality indexes: {e}")
    
    def _create_session_indexes(self):
        """Create indexes for user sessions"""
        try:
            self.user_sessions.create_index([
                ("session_id", ASCENDING)
            ], unique=True, name="session_id_unique_idx")
            
            self.user_sessions.create_index([
                ("created_at", DESCENDING)
            ], name="session_created_idx")
            
            # TTL index for session cleanup (keep for 24 hours)
            self.user_sessions.create_index([
                ("created_at", ASCENDING)
            ], expireAfterSeconds=86400, name="session_ttl_idx")  # 24 hours
            
            logger.info("User session indexes created")
            
        except Exception as e:
            logger.error(f"Error creating session indexes: {e}")
    
    def store_market_data_bulk(self, data_points: List[Dict]) -> int:
        """Store market data in bulk with deduplication"""
        try:
            if not data_points:
                return 0
            
            inserted_count = 0
            
            # Use ordered=False to continue on duplicate key errors
            try:
                result = self.market_data.insert_many(data_points, ordered=False)
                inserted_count = len(result.inserted_ids)
            except Exception as e:
                if "duplicate key error" in str(e).lower():
                    # Count successful insertions from the error details
                    error_details = str(e)
                    # This is a simplified approach - in production, you'd parse the error more carefully
                    inserted_count = len(data_points) - error_details.count("duplicate key")
                    logger.info(f"Bulk insert completed with {inserted_count} new records, duplicates skipped")
                else:
                    raise e
            
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error in bulk market data storage: {e}")
            return 0
    
    def get_market_data_range(self, symbol: str, start_date: datetime, 
                            end_date: datetime) -> Optional[pd.DataFrame]:
        """Retrieve market data for a symbol within date range"""
        try:
            query = {
                'symbol': symbol,
                'timestamp': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
            
            cursor = self.market_data.find(query).sort('timestamp', ASCENDING)
            data = list(cursor)
            
            if not data:
                return None
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Select relevant columns
            columns = ['open', 'high', 'low', 'close', 'volume', 'adjusted_close']
            available_columns = [col for col in columns if col in df.columns]
            
            return df[available_columns]
            
        except Exception as e:
            logger.error(f"Error retrieving market data: {e}")
            return None
    
    def store_portfolio_analysis(self, analysis_data: Dict) -> str:
        """Store portfolio analysis results with duplicate prevention"""
        try:
            # Create unique hash for deduplication
            portfolio_hash = self._create_portfolio_hash(analysis_data)

            # Check if similar analysis already exists (within last hour)
            one_hour_ago = datetime.now() - timedelta(hours=1)
            existing = self.portfolio_analyses.find_one({
                'portfolio_hash': portfolio_hash,
                'created_at': {'$gte': one_hour_ago}
            })

            if existing:
                logger.info(f"🔄 Similar analysis found, returning existing ID: {existing['_id']}")
                return str(existing['_id'])

            # Add metadata for new analysis
            analysis_data['created_at'] = datetime.now()
            analysis_data['_id'] = ObjectId()
            analysis_data['portfolio_hash'] = portfolio_hash

            result = self.portfolio_analyses.insert_one(analysis_data)

            logger.info(f"✅ New portfolio analysis stored with ID: {result.inserted_id}")
            return str(result.inserted_id)

        except Exception as e:
            logger.error(f"Error storing portfolio analysis: {e}")
            return ""

    def _create_portfolio_hash(self, analysis_data: Dict) -> str:
        """Create unique hash for portfolio to prevent duplicates"""
        try:
            import hashlib

            # Extract key portfolio characteristics
            portfolio_key = {
                'stocks': sorted(analysis_data.get('portfolio', {}).get('stocks', [])),
                'weights': analysis_data.get('portfolio', {}).get('weights', []),
                'period': analysis_data.get('period', '1y'),
                'risk_free_rate': analysis_data.get('risk_free_rate', 0.02)
            }

            # Create hash
            portfolio_str = str(sorted(portfolio_key.items()))
            return hashlib.md5(portfolio_str.encode()).hexdigest()

        except Exception as e:
            logger.error(f"Error creating portfolio hash: {e}")
            return str(ObjectId())  # Fallback to unique ID
    
    def get_portfolio_analysis(self, analysis_id: str) -> Optional[Dict]:
        """Retrieve portfolio analysis by ID"""
        try:
            result = self.portfolio_analyses.find_one({'_id': ObjectId(analysis_id)})
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving portfolio analysis: {e}")
            return None
    
    def store_optimization_result(self, optimization_data: Dict) -> str:
        """Store portfolio optimization results"""
        try:
            optimization_data['created_at'] = datetime.now()
            optimization_data['_id'] = ObjectId()
            
            result = self.optimization_results.insert_one(optimization_data)
            
            logger.info(f"Optimization result stored with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error storing optimization result: {e}")
            return ""
    
    def get_data_quality_stats(self, symbol: str = None) -> Dict:
        """Get data quality statistics"""
        try:
            pipeline = []
            
            if symbol:
                pipeline.append({'$match': {'symbol': symbol}})
            
            pipeline.extend([
                {
                    '$group': {
                        '_id': '$symbol',
                        'total_records': {'$sum': 1},
                        'avg_volume': {'$avg': '$volume'},
                        'latest_timestamp': {'$max': '$timestamp'},
                        'earliest_timestamp': {'$min': '$timestamp'},
                        'sources': {'$addToSet': '$source'}
                    }
                },
                {
                    '$sort': {'total_records': -1}
                }
            ])
            
            results = list(self.market_data.aggregate(pipeline))
            
            return {
                'symbols_count': len(results),
                'symbol_stats': results,
                'generated_at': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting data quality stats: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data beyond retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean up old market data
            market_result = self.market_data.delete_many({
                'timestamp': {'$lt': cutoff_date}
            })
            
            logger.info(f"Cleaned up {market_result.deleted_count} old market data records")
            
            return {
                'market_data_deleted': market_result.deleted_count,
                'cutoff_date': cutoff_date
            }
            
        except Exception as e:
            logger.error(f"Error during data cleanup: {e}")
            return {}
    
    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics"""
        try:
            stats = {}
            
            # Collection stats
            for collection_name in self.db.list_collection_names():
                collection = self.db[collection_name]
                stats[collection_name] = {
                    'document_count': collection.count_documents({}),
                    'indexes': len(list(collection.list_indexes())),
                    'size_mb': self.db.command('collStats', collection_name).get('size', 0) / (1024 * 1024)
                }
            
            # Database size
            db_stats = self.db.command('dbStats')
            stats['database'] = {
                'total_size_mb': db_stats.get('dataSize', 0) / (1024 * 1024),
                'index_size_mb': db_stats.get('indexSize', 0) / (1024 * 1024),
                'collections_count': db_stats.get('collections', 0)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def close_connection(self):
        """Close database connection"""
        try:
            self.client.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
