"""
Portfolio Optimization Module - Modern Portfolio Theory implementation
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Portfolio optimization using Modern Portfolio Theory
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.trading_days = 252
    
    def optimize_portfolio(self, prices: pd.DataFrame, objective: str = 'max_sharpe') -> Dict:
        """
        Optimize portfolio based on specified objective
        
        Args:
            prices: DataFrame with historical prices
            objective: 'max_sharpe', 'min_risk', or 'max_return'
        
        Returns:
            Dictionary with optimization results
        """
        try:
            # Validate input data
            if prices.empty or len(prices.columns) == 0:
                logger.warning("Empty price data provided for optimization")
                return self._get_default_optimization_result()

            # Handle single asset case
            if len(prices.columns) == 1:
                symbol = prices.columns[0]
                returns = prices.pct_change().dropna()
                if returns.empty:
                    return self._get_default_optimization_result()

                annual_return = returns.mean().iloc[0] * self.trading_days
                annual_vol = returns.std().iloc[0] * np.sqrt(self.trading_days)
                sharpe = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0

                return {
                    'weights': [1.0],
                    'symbols': [symbol],
                    'metrics': {
                        'expected_return': float(annual_return),
                        'volatility': float(annual_vol),
                        'sharpe_ratio': float(sharpe)
                    },
                    'success': True,
                    'message': 'Single asset portfolio - no optimization needed'
                }

            # Calculate returns and statistics
            returns = prices.pct_change().dropna()

            # Check for sufficient data
            if returns.empty or len(returns) < 2:
                logger.warning("Insufficient return data for optimization")
                return self._get_default_optimization_result()

            mean_returns = returns.mean() * self.trading_days
            cov_matrix = returns.cov() * self.trading_days

            # Handle NaN values
            if mean_returns.isna().any() or cov_matrix.isna().any().any():
                logger.warning("NaN values found in return calculations")
                return self._get_default_optimization_result()

            num_assets = len(prices.columns)

            # Constraints and bounds
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
            bounds = tuple((0, 1) for _ in range(num_assets))  # Long-only portfolio

            # Initial guess (equal weights)
            initial_guess = np.array([1/num_assets] * num_assets)
            
            results = {}
            
            # Maximum Sharpe Ratio optimization
            if objective in ['max_sharpe', 'all']:
                max_sharpe_result = self._optimize_sharpe(
                    mean_returns, cov_matrix, constraints, bounds, initial_guess
                )
                results['max_sharpe'] = max_sharpe_result
            
            # Minimum Risk optimization
            if objective in ['min_risk', 'all']:
                min_risk_result = self._optimize_risk(
                    mean_returns, cov_matrix, constraints, bounds, initial_guess
                )
                results['min_risk'] = min_risk_result
            
            # Maximum Return optimization (for given risk level)
            if objective in ['max_return', 'all']:
                max_return_result = self._optimize_return(
                    mean_returns, cov_matrix, constraints, bounds, initial_guess
                )
                results['max_return'] = max_return_result
            
            # Generate efficient frontier
            if objective == 'all':
                efficient_frontier = self._generate_efficient_frontier(
                    mean_returns, cov_matrix, constraints, bounds
                )
                results['efficient_frontier'] = efficient_frontier
            
            # Add current portfolio comparison if we have equal weights
            equal_weights = [1/num_assets] * num_assets
            equal_weight_metrics = self._calculate_portfolio_metrics(
                equal_weights, mean_returns, cov_matrix
            )
            results['equal_weight_benchmark'] = {
                'weights': equal_weights,
                'metrics': equal_weight_metrics,
                'symbols': list(prices.columns)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return self._get_default_optimization_result()

    def _get_default_optimization_result(self):
        """Return default optimization result for error cases"""
        return {
            'weights': [1.0],
            'symbols': ['UNKNOWN'],
            'metrics': {
                'expected_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0
            },
            'success': False,
            'message': 'Optimization failed - insufficient data'
        }
    
    def _optimize_sharpe(self, mean_returns, cov_matrix, constraints, bounds, initial_guess):
        """Optimize for maximum Sharpe ratio"""
        def negative_sharpe(weights):
            weights = np.array(weights)  # Ensure numpy array
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std
            return -sharpe  # Negative because we minimize
        
        result = minimize(
            negative_sharpe, initial_guess, method='SLSQP',
            bounds=bounds, constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            metrics = self._calculate_portfolio_metrics(optimal_weights, mean_returns, cov_matrix)
            
            return {
                'weights': optimal_weights.tolist(),
                'metrics': metrics,
                'success': True,
                'message': 'Optimization successful'
            }
        else:
            return {
                'success': False,
                'message': f'Optimization failed: {result.message}'
            }
    
    def _optimize_risk(self, mean_returns, cov_matrix, constraints, bounds, initial_guess):
        """Optimize for minimum risk (variance)"""
        def portfolio_variance(weights):
            weights = np.array(weights)  # Ensure numpy array
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        result = minimize(
            portfolio_variance, initial_guess, method='SLSQP',
            bounds=bounds, constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            metrics = self._calculate_portfolio_metrics(optimal_weights, mean_returns, cov_matrix)
            
            return {
                'weights': optimal_weights.tolist(),
                'metrics': metrics,
                'success': True,
                'message': 'Optimization successful'
            }
        else:
            return {
                'success': False,
                'message': f'Optimization failed: {result.message}'
            }
    
    def _optimize_return(self, mean_returns, cov_matrix, constraints, bounds, initial_guess):
        """Optimize for maximum return (with risk constraint)"""
        # Target risk level (e.g., 15% annual volatility)
        target_risk = 0.15
        
        def negative_return(weights):
            return -np.sum(mean_returns * weights)
        
        # Add risk constraint
        risk_constraint = {
            'type': 'ineq',
            'fun': lambda x: target_risk - np.sqrt(np.dot(x.T, np.dot(cov_matrix, x)))
        }
        
        all_constraints = [constraints, risk_constraint]
        
        result = minimize(
            negative_return, initial_guess, method='SLSQP',
            bounds=bounds, constraints=all_constraints
        )
        
        if result.success:
            optimal_weights = result.x
            metrics = self._calculate_portfolio_metrics(optimal_weights, mean_returns, cov_matrix)
            
            return {
                'weights': optimal_weights.tolist(),
                'metrics': metrics,
                'target_risk': target_risk,
                'success': True,
                'message': 'Optimization successful'
            }
        else:
            return {
                'success': False,
                'message': f'Optimization failed: {result.message}'
            }
    
    def _generate_efficient_frontier(self, mean_returns, cov_matrix, constraints, bounds, num_points=50):
        """Generate efficient frontier points"""
        min_ret = mean_returns.min()
        max_ret = mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, num_points)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            # Add return constraint
            return_constraint = {
                'type': 'eq',
                'fun': lambda x, target=target_return: np.sum(mean_returns * x) - target
            }
            
            all_constraints = [constraints, return_constraint]
            
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            initial_guess = np.array([1/len(mean_returns)] * len(mean_returns))
            
            result = minimize(
                portfolio_variance, initial_guess, method='SLSQP',
                bounds=bounds, constraints=all_constraints
            )
            
            if result.success:
                weights = result.x
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe = (portfolio_return - self.risk_free_rate) / portfolio_risk
                
                efficient_portfolios.append({
                    'return': float(portfolio_return),
                    'risk': float(portfolio_risk),
                    'sharpe': float(sharpe),
                    'weights': weights.tolist()
                })
        
        return efficient_portfolios
    
    def _calculate_portfolio_metrics(self, weights, mean_returns, cov_matrix):
        """Calculate portfolio metrics for given weights"""
        weights = np.array(weights)  # Ensure numpy array
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
        
        return {
            'expected_return': float(portfolio_return),
            'volatility': float(portfolio_std),
            'sharpe_ratio': float(sharpe_ratio),
            'variance': float(portfolio_variance)
        }
    
    def calculate_risk_parity_weights(self, prices: pd.DataFrame) -> Dict:
        """
        Calculate risk parity (equal risk contribution) weights
        """
        try:
            returns = prices.pct_change().dropna()
            cov_matrix = returns.cov() * self.trading_days
            
            num_assets = len(prices.columns)
            
            def risk_budget_objective(weights):
                """Objective function for risk parity"""
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                
                # We want equal risk contributions
                target_contrib = portfolio_vol / num_assets
                return np.sum((contrib - target_contrib) ** 2)
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0.001, 1) for _ in range(num_assets))  # Small minimum to avoid division by zero
            initial_guess = np.array([1/num_assets] * num_assets)
            
            result = minimize(
                risk_budget_objective, initial_guess, method='SLSQP',
                bounds=bounds, constraints=constraints
            )
            
            if result.success:
                weights = result.x
                mean_returns = returns.mean() * self.trading_days
                metrics = self._calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
                
                return {
                    'weights': weights.tolist(),
                    'metrics': metrics,
                    'symbols': list(prices.columns),
                    'success': True,
                    'message': 'Risk parity optimization successful'
                }
            else:
                return {
                    'success': False,
                    'message': f'Risk parity optimization failed: {result.message}'
                }
                
        except Exception as e:
            logger.error(f"Error in risk parity calculation: {e}")
            return {'error': str(e), 'success': False}
