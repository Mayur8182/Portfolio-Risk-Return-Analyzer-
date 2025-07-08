#!/usr/bin/env python3
"""
Speed Test Script for Enhanced Data Fetcher
Tests the performance improvements in data fetching
"""

import time
import logging
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_data_fetcher import EnhancedDataFetcher
from simple_data_fetcher import SimpleDataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_fetcher_speed():
    """Test the enhanced data fetcher speed"""
    logger.info("🚀 Testing Enhanced Data Fetcher Speed")
    
    # Test symbols (reduced for faster testing)
    test_symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    try:
        # Initialize enhanced fetcher
        enhanced_fetcher = EnhancedDataFetcher()
        
        # Test 1: Fast async method
        logger.info("Test 1: Fast Async Portfolio Data Fetching")
        start_time = time.time()
        
        portfolio_data = enhanced_fetcher.get_portfolio_data_enhanced(test_symbols)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if portfolio_data is not None and not portfolio_data.empty:
            logger.info(f"✅ Enhanced fetcher completed in {elapsed_time:.2f} seconds")
            logger.info(f"📊 Data shape: {portfolio_data.shape}")
            logger.info(f"📈 Symbols fetched: {list(portfolio_data.columns)}")
            logger.info(f"⚡ Speed: {len(portfolio_data) * len(portfolio_data.columns) / elapsed_time:.0f} data points/second")
        else:
            logger.error("❌ Enhanced fetcher failed to fetch data")
            
        # Test 2: Cache performance
        logger.info("\nTest 2: Cache Performance Test")
        start_time = time.time()
        
        # Second call should be much faster due to caching
        portfolio_data_cached = enhanced_fetcher.get_portfolio_data_enhanced(test_symbols)
        
        end_time = time.time()
        cached_elapsed_time = end_time - start_time
        
        if portfolio_data_cached is not None and not portfolio_data_cached.empty:
            logger.info(f"✅ Cached fetch completed in {cached_elapsed_time:.2f} seconds")
            logger.info(f"🚀 Speed improvement: {elapsed_time / cached_elapsed_time:.1f}x faster")
        
        # Cleanup
        enhanced_fetcher.cleanup()
        
        return elapsed_time, cached_elapsed_time
        
    except Exception as e:
        logger.error(f"❌ Enhanced fetcher test failed: {e}")
        return None, None

def test_simple_fetcher_speed():
    """Test the simple data fetcher speed for comparison"""
    logger.info("\n📊 Testing Simple Data Fetcher Speed (for comparison)")

    # Test symbols
    test_symbols = ['AAPL', 'GOOGL', 'MSFT']  # Reduced for faster testing

    try:
        # Initialize simple fetcher
        simple_fetcher = SimpleDataFetcher()

        start_time = time.time()

        # Use the correct method name
        portfolio_data = simple_fetcher.get_data(test_symbols)

        end_time = time.time()
        elapsed_time = end_time - start_time

        if portfolio_data is not None and not portfolio_data.empty:
            logger.info(f"✅ Simple fetcher completed in {elapsed_time:.2f} seconds")
            logger.info(f"📊 Data shape: {portfolio_data.shape}")
            logger.info(f"📈 Symbols fetched: {list(portfolio_data.columns)}")
            logger.info(f"⚡ Speed: {len(portfolio_data) * len(portfolio_data.columns) / elapsed_time:.0f} data points/second")
        else:
            logger.error("❌ Simple fetcher failed to fetch data")

        return elapsed_time

    except Exception as e:
        logger.error(f"❌ Simple fetcher test failed: {e}")
        return None

def main():
    """Main test function"""
    logger.info("🔥 Portfolio Analytics - Data Fetching Speed Test")
    logger.info("=" * 60)
    
    # Test enhanced fetcher
    enhanced_time, cached_time = test_enhanced_fetcher_speed()
    
    # Test simple fetcher
    simple_time = test_simple_fetcher_speed()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📈 PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    
    if enhanced_time and simple_time:
        improvement = simple_time / enhanced_time
        logger.info(f"🚀 Enhanced Fetcher: {enhanced_time:.2f}s")
        logger.info(f"📊 Simple Fetcher: {simple_time:.2f}s")
        logger.info(f"⚡ Speed Improvement: {improvement:.1f}x faster")
        
        if cached_time:
            cache_improvement = enhanced_time / cached_time
            logger.info(f"💾 Cached Fetch: {cached_time:.2f}s")
            logger.info(f"🚀 Cache Speed Boost: {cache_improvement:.1f}x faster")
    
    logger.info("\n✅ Speed test completed!")

if __name__ == "__main__":
    main()
