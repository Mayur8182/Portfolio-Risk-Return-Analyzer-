"""
Portfolio Risk Analysis Module - Core financial calculations and risk metrics
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PortfolioAnalyzer:
    """
    Comprehensive portfolio risk and performance analysis
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize analyzer with risk-free rate (default 2% annually)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = 252  # Approximate trading days per year
    
    def calculate_returns(self, prices: pd.DataFrame, weights: List[float]) -> pd.Series:
        """
        Calculate portfolio returns given prices and weights
        """
        # Calculate daily returns for each asset
        returns = prices.pct_change().dropna()

        # Debug logging
        logger.info(f"Returns DataFrame shape: {returns.shape}, columns: {list(returns.columns)}")
        logger.info(f"Weights length: {len(weights)}, weights: {weights}")

        # Ensure weights match the number of columns
        if len(weights) != len(returns.columns):
            logger.error(f"Mismatch: {len(weights)} weights for {len(returns.columns)} assets")
            # Adjust weights to match columns
            if len(weights) > len(returns.columns):
                weights = weights[:len(returns.columns)]
                logger.warning(f"Truncated weights to match assets: {weights}")
            else:
                # Pad with zeros if fewer weights
                weights = weights + [0.0] * (len(returns.columns) - len(weights))
                logger.warning(f"Padded weights with zeros: {weights}")

        # Convert weights to numpy array for proper broadcasting
        weights_array = np.array(weights)

        # Calculate weighted portfolio returns
        portfolio_returns = (returns * weights_array).sum(axis=1)

        return portfolio_returns
    
    def calculate_risk_metrics(self, prices: pd.DataFrame, weights: List[float]) -> Dict:
        """
        Calculate comprehensive risk metrics for the portfolio
        """
        try:
            portfolio_returns = self.calculate_returns(prices, weights)
            
            # Basic statistics
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std()
            
            # Annualized metrics
            annualized_return = (1 + mean_return) ** self.trading_days - 1
            annualized_volatility = std_return * np.sqrt(self.trading_days)
            
            # Sharpe Ratio
            sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility
            
            # Value at Risk (VaR)
            var_95 = np.percentile(portfolio_returns, 5)
            var_99 = np.percentile(portfolio_returns, 1)
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
            
            # Maximum Drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Sortino Ratio (downside deviation)
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(self.trading_days)
            sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Skewness and Kurtosis
            skewness = stats.skew(portfolio_returns)
            kurtosis = stats.kurtosis(portfolio_returns)
            
            # Beta calculation (using SPY as market proxy)
            beta = self._calculate_beta(portfolio_returns)
            
            # Calmar Ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            return {
                'daily_return_mean': float(mean_return),
                'daily_return_std': float(std_return),
                'annualized_return': float(annualized_return),
                'annualized_volatility': float(annualized_volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'calmar_ratio': float(calmar_ratio),
                'var_95': float(var_95),
                'var_99': float(var_99),
                'cvar_95': float(cvar_95),
                'cvar_99': float(cvar_99),
                'max_drawdown': float(max_drawdown),
                'beta': float(beta) if beta is not None else None,
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'total_return': float((cumulative_returns.iloc[-1] - 1)),
                'volatility_rank': self._get_volatility_rank(annualized_volatility)
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def calculate_performance_metrics(self, prices: pd.DataFrame, weights: List[float]) -> Dict:
        """
        Calculate performance attribution and metrics
        """
        try:
            portfolio_returns = self.calculate_returns(prices, weights)
            individual_returns = prices.pct_change().dropna()
            
            # Portfolio performance
            cumulative_portfolio = (1 + portfolio_returns).cumprod()
            
            # Individual asset performance
            individual_performance = {}
            for i, symbol in enumerate(prices.columns):
                asset_returns = individual_returns[symbol]
                cumulative_asset = (1 + asset_returns).cumprod()
                
                individual_performance[symbol] = {
                    'weight': weights[i],
                    'total_return': float(cumulative_asset.iloc[-1] - 1),
                    'annualized_return': float((1 + asset_returns.mean()) ** self.trading_days - 1),
                    'volatility': float(asset_returns.std() * np.sqrt(self.trading_days)),
                    'sharpe_ratio': float((asset_returns.mean() * self.trading_days - self.risk_free_rate) / 
                                        (asset_returns.std() * np.sqrt(self.trading_days))),
                    'contribution_to_return': float(weights[i] * asset_returns.mean() * self.trading_days),
                    'contribution_to_risk': float(weights[i] ** 2 * (asset_returns.std() * np.sqrt(self.trading_days)) ** 2)
                }
            
            # Performance attribution
            total_contribution_return = sum([perf['contribution_to_return'] for perf in individual_performance.values()])
            total_contribution_risk = sum([perf['contribution_to_risk'] for perf in individual_performance.values()])
            
            return {
                'portfolio_total_return': float(cumulative_portfolio.iloc[-1] - 1),
                'portfolio_annualized_return': float((1 + portfolio_returns.mean()) ** self.trading_days - 1),
                'individual_performance': individual_performance,
                'attribution': {
                    'total_return_contribution': float(total_contribution_return),
                    'total_risk_contribution': float(total_contribution_risk)
                },
                'best_performer': max(individual_performance.keys(), 
                                    key=lambda x: individual_performance[x]['total_return']),
                'worst_performer': min(individual_performance.keys(), 
                                     key=lambda x: individual_performance[x]['total_return']),
                'most_volatile': max(individual_performance.keys(), 
                                   key=lambda x: individual_performance[x]['volatility']),
                'least_volatile': min(individual_performance.keys(), 
                                    key=lambda x: individual_performance[x]['volatility'])
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def calculate_correlation_matrix(self, prices: pd.DataFrame) -> Dict:
        """
        Calculate correlation matrix and related statistics
        """
        try:
            # Validate input data
            if prices.empty or len(prices.columns) == 0:
                logger.warning("Empty price data provided for correlation calculation")
                return self._get_default_correlation_matrix()

            returns = prices.pct_change().dropna()

            # Check if we have sufficient data
            if returns.empty or len(returns) < 2:
                logger.warning("Insufficient return data for correlation calculation")
                return self._get_default_correlation_matrix()

            # Handle single asset case
            if len(returns.columns) == 1:
                symbol = returns.columns[0]
                return {
                    'matrix': {symbol: {symbol: 1.0}},
                    'statistics': {
                        'average_correlation': 1.0,
                        'max_correlation': 1.0,
                        'min_correlation': 1.0,
                        'most_correlated_pair': (symbol, symbol),
                        'least_correlated_pair': (symbol, symbol)
                    },
                    'diversification_ratio': 1.0
                }

            correlation_matrix = returns.corr()

            # Handle NaN values
            correlation_matrix = correlation_matrix.fillna(0)

            # Convert to dictionary format
            corr_dict = correlation_matrix.to_dict()

            # Calculate average correlations
            correlations = []
            symbols = list(correlation_matrix.columns)

            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if not np.isnan(corr_val):
                        correlations.append(corr_val)

            # Handle case where no valid correlations exist
            if not correlations:
                logger.warning("No valid correlations found")
                return self._get_default_correlation_matrix()

            avg_correlation = np.mean(correlations)
            max_correlation = np.max(correlations)
            min_correlation = np.min(correlations)
            
            # Find most and least correlated pairs
            max_pair = None
            min_pair = None
            
            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if corr_val == max_correlation:
                        max_pair = (symbols[i], symbols[j])
                    if corr_val == min_correlation:
                        min_pair = (symbols[i], symbols[j])
            
            return {
                'matrix': corr_dict,
                'statistics': {
                    'average_correlation': float(avg_correlation),
                    'max_correlation': float(max_correlation),
                    'min_correlation': float(min_correlation),
                    'most_correlated_pair': max_pair,
                    'least_correlated_pair': min_pair
                },
                'diversification_ratio': self._calculate_diversification_ratio(returns)
            }
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return self._get_default_correlation_matrix()

    def _get_default_correlation_matrix(self):
        """Return default correlation matrix for error cases"""
        return {
            'matrix': {},
            'statistics': {
                'average_correlation': 0.0,
                'max_correlation': 0.0,
                'min_correlation': 0.0,
                'most_correlated_pair': None,
                'least_correlated_pair': None
            },
            'diversification_ratio': 1.0
        }
    
    def _calculate_beta(self, portfolio_returns: pd.Series) -> Optional[float]:
        """
        Calculate portfolio beta against market (SPY)
        """
        try:
            # This is a simplified beta calculation
            # In a real implementation, you'd fetch SPY data and calculate covariance
            # For now, return a placeholder
            return 1.0
        except Exception:
            return None
    
    def _get_volatility_rank(self, volatility: float) -> str:
        """
        Classify volatility level
        """
        if volatility < 0.10:
            return "Low"
        elif volatility < 0.20:
            return "Moderate"
        elif volatility < 0.30:
            return "High"
        else:
            return "Very High"
    
    def _calculate_diversification_ratio(self, returns: pd.DataFrame) -> float:
        """
        Calculate diversification ratio
        """
        try:
            # Weighted average of individual volatilities
            individual_vols = returns.std()
            weights = [1/len(returns.columns)] * len(returns.columns)  # Equal weights for simplicity
            weighted_avg_vol = sum(w * vol for w, vol in zip(weights, individual_vols))
            
            # Portfolio volatility
            portfolio_vol = (returns * weights).sum(axis=1).std()
            
            return float(weighted_avg_vol / portfolio_vol) if portfolio_vol > 0 else 1.0
        except Exception:
            return 1.0
