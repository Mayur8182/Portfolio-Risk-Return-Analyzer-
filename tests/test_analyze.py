"""
Unit Tests for Portfolio Analysis Module
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from analyze import PortfolioAnalyzer
from fetch_data import PortfolioDataFetcher
from optimize import PortfolioOptimizer
from sentiment import SentimentAnalyzer
from utils import validate_portfolio_input, format_currency, format_percentage

class TestPortfolioAnalyzer(unittest.TestCase):
    """Test cases for PortfolioAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = PortfolioAnalyzer()
        
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)  # For reproducible tests
        
        # Generate sample price data for 3 stocks
        self.sample_prices = pd.DataFrame({
            'AAPL': 150 + np.cumsum(np.random.normal(0, 2, len(dates))),
            'GOOGL': 2500 + np.cumsum(np.random.normal(0, 30, len(dates))),
            'MSFT': 300 + np.cumsum(np.random.normal(0, 5, len(dates)))
        }, index=dates)
        
        self.sample_weights = [0.4, 0.35, 0.25]
    
    def test_calculate_returns(self):
        """Test portfolio returns calculation"""
        returns = self.analyzer.calculate_returns(self.sample_prices, self.sample_weights)
        
        # Check that returns is a pandas Series
        self.assertIsInstance(returns, pd.Series)
        
        # Check that returns has the correct length (one less than prices due to pct_change)
        self.assertEqual(len(returns), len(self.sample_prices) - 1)
        
        # Check that returns are reasonable (between -50% and +50% daily)
        self.assertTrue(returns.min() > -0.5)
        self.assertTrue(returns.max() < 0.5)
    
    def test_calculate_risk_metrics(self):
        """Test risk metrics calculation"""
        risk_metrics = self.analyzer.calculate_risk_metrics(self.sample_prices, self.sample_weights)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'daily_return_mean', 'daily_return_std', 'annualized_return',
            'annualized_volatility', 'sharpe_ratio', 'sortino_ratio',
            'var_95', 'var_99', 'max_drawdown', 'skewness', 'kurtosis'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, risk_metrics)
            self.assertIsInstance(risk_metrics[metric], (int, float))
        
        # Check that volatility is positive
        self.assertGreater(risk_metrics['annualized_volatility'], 0)
        
        # Check that VaR values are negative (representing losses)
        self.assertLess(risk_metrics['var_95'], 0)
        self.assertLess(risk_metrics['var_99'], 0)
        
        # Check that 99% VaR is more extreme than 95% VaR
        self.assertLess(risk_metrics['var_99'], risk_metrics['var_95'])
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation"""
        performance_metrics = self.analyzer.calculate_performance_metrics(
            self.sample_prices, self.sample_weights
        )
        
        # Check that all expected keys are present
        expected_keys = [
            'portfolio_total_return', 'portfolio_annualized_return',
            'individual_performance', 'attribution'
        ]
        
        for key in expected_keys:
            self.assertIn(key, performance_metrics)
        
        # Check individual performance for each stock
        individual_perf = performance_metrics['individual_performance']
        for i, symbol in enumerate(self.sample_prices.columns):
            self.assertIn(symbol, individual_perf)
            self.assertEqual(individual_perf[symbol]['weight'], self.sample_weights[i])
    
    def test_calculate_correlation_matrix(self):
        """Test correlation matrix calculation"""
        correlation_data = self.analyzer.calculate_correlation_matrix(self.sample_prices)
        
        # Check that matrix key exists
        self.assertIn('matrix', correlation_data)
        self.assertIn('statistics', correlation_data)
        
        # Check matrix dimensions
        matrix = correlation_data['matrix']
        symbols = list(self.sample_prices.columns)
        
        for symbol in symbols:
            self.assertIn(symbol, matrix)
            self.assertEqual(len(matrix[symbol]), len(symbols))
        
        # Check that diagonal elements are 1 (correlation with self)
        for symbol in symbols:
            self.assertAlmostEqual(matrix[symbol][symbol], 1.0, places=5)


class TestPortfolioOptimizer(unittest.TestCase):
    """Test cases for PortfolioOptimizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = PortfolioOptimizer()
        
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        self.sample_prices = pd.DataFrame({
            'AAPL': 150 + np.cumsum(np.random.normal(0.001, 0.02, len(dates))),
            'GOOGL': 2500 + np.cumsum(np.random.normal(0.0008, 0.025, len(dates))),
            'MSFT': 300 + np.cumsum(np.random.normal(0.0012, 0.018, len(dates)))
        }, index=dates)
    
    def test_optimize_portfolio_max_sharpe(self):
        """Test maximum Sharpe ratio optimization"""
        result = self.optimizer.optimize_portfolio(self.sample_prices, objective='max_sharpe')
        
        # Check that optimization was successful
        self.assertIn('max_sharpe', result)
        max_sharpe_result = result['max_sharpe']
        
        if max_sharpe_result.get('success'):
            # Check that weights sum to 1
            weights = max_sharpe_result['weights']
            self.assertAlmostEqual(sum(weights), 1.0, places=5)
            
            # Check that all weights are non-negative (long-only constraint)
            for weight in weights:
                self.assertGreaterEqual(weight, 0)
            
            # Check that metrics are present
            metrics = max_sharpe_result['metrics']
            self.assertIn('expected_return', metrics)
            self.assertIn('volatility', metrics)
            self.assertIn('sharpe_ratio', metrics)
    
    def test_optimize_portfolio_min_risk(self):
        """Test minimum risk optimization"""
        result = self.optimizer.optimize_portfolio(self.sample_prices, objective='min_risk')
        
        # Check that optimization was successful
        self.assertIn('min_risk', result)
        min_risk_result = result['min_risk']
        
        if min_risk_result.get('success'):
            # Check that weights sum to 1
            weights = min_risk_result['weights']
            self.assertAlmostEqual(sum(weights), 1.0, places=5)
            
            # Check that all weights are non-negative
            for weight in weights:
                self.assertGreaterEqual(weight, 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""
    
    def test_validate_portfolio_input_valid(self):
        """Test portfolio input validation with valid data"""
        valid_input = {
            'stocks': ['AAPL', 'GOOGL', 'MSFT'],
            'weights': [0.4, 0.35, 0.25]
        }
        
        result = validate_portfolio_input(valid_input)
        self.assertTrue(result['valid'])
    
    def test_validate_portfolio_input_invalid_weights(self):
        """Test portfolio input validation with invalid weights"""
        invalid_input = {
            'stocks': ['AAPL', 'GOOGL', 'MSFT'],
            'weights': [0.4, 0.35, 0.35]  # Sum > 1
        }
        
        result = validate_portfolio_input(invalid_input)
        self.assertFalse(result['valid'])
    
    def test_validate_portfolio_input_mismatched_lengths(self):
        """Test portfolio input validation with mismatched lengths"""
        invalid_input = {
            'stocks': ['AAPL', 'GOOGL'],
            'weights': [0.4, 0.35, 0.25]  # Different length
        }
        
        result = validate_portfolio_input(invalid_input)
        self.assertFalse(result['valid'])
    
    def test_format_currency(self):
        """Test currency formatting"""
        self.assertEqual(format_currency(1000), '$1.00K')
        self.assertEqual(format_currency(1000000), '$1.00M')
        self.assertEqual(format_currency(1000000000), '$1.00B')
        self.assertEqual(format_currency(123.45), '$123.45')
    
    def test_format_percentage(self):
        """Test percentage formatting"""
        self.assertEqual(format_percentage(0.1234), '12.34%')
        self.assertEqual(format_percentage(0.1234, 1), '12.3%')
        self.assertEqual(format_percentage(-0.05), '-5.00%')


class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for SentimentAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def test_analyze_stock_sentiment(self):
        """Test sentiment analysis for a single stock"""
        result = self.sentiment_analyzer.analyze_stock_sentiment('AAPL')
        
        # Check that all expected keys are present
        expected_keys = ['symbol', 'overall_score', 'sentiment_label', 'components', 'confidence']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check that sentiment score is in valid range
        self.assertGreaterEqual(result['overall_score'], -1)
        self.assertLessEqual(result['overall_score'], 1)
        
        # Check that components exist
        components = result['components']
        self.assertIn('news', components)
        self.assertIn('social', components)
        self.assertIn('technical', components)
    
    def test_analyze_portfolio_sentiment(self):
        """Test sentiment analysis for a portfolio"""
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        result = self.sentiment_analyzer.analyze_portfolio_sentiment(symbols)
        
        # Check that all expected keys are present
        expected_keys = ['individual_sentiment', 'portfolio_sentiment', 'sentiment_distribution']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check individual sentiment for each symbol
        individual_sentiment = result['individual_sentiment']
        for symbol in symbols:
            self.assertIn(symbol, individual_sentiment)


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestPortfolioAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestPortfolioOptimizer))
    test_suite.addTest(unittest.makeSuite(TestUtilityFunctions))
    test_suite.addTest(unittest.makeSuite(TestSentimentAnalyzer))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)
