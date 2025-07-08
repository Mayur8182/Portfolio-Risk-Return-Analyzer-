"""
Chart Data Generator - Prepares data for frontend visualizations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class ChartDataGenerator:
    """
    Generate data structures optimized for frontend chart libraries (Chart.js, D3.js)
    """
    
    def __init__(self):
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
    
    def generate_all_charts(self, market_data: pd.DataFrame, weights: List[float], 
                          risk_metrics: Dict, performance_metrics: Dict) -> Dict:
        """
        Generate all chart data for the dashboard
        
        Args:
            market_data: Historical price data
            weights: Portfolio weights
            risk_metrics: Risk analysis results
            performance_metrics: Performance analysis results
        
        Returns:
            Dictionary containing all chart data
        """
        try:
            charts = {}
            
            # Portfolio performance chart
            charts['portfolio_performance'] = self.generate_portfolio_performance_chart(
                market_data, weights
            )
            
            # Asset allocation pie chart
            charts['asset_allocation'] = self.generate_asset_allocation_chart(
                list(market_data.columns), weights
            )
            
            # Risk metrics radar chart
            charts['risk_radar'] = self.generate_risk_radar_chart(risk_metrics)
            
            # Correlation heatmap
            charts['correlation_heatmap'] = self.generate_correlation_heatmap(market_data)
            
            # Individual asset performance
            charts['individual_performance'] = self.generate_individual_performance_chart(
                market_data, performance_metrics
            )
            
            # Volatility comparison
            charts['volatility_comparison'] = self.generate_volatility_comparison_chart(
                market_data, performance_metrics
            )
            
            # Returns distribution
            charts['returns_distribution'] = self.generate_returns_distribution_chart(
                market_data, weights
            )
            
            # Risk contribution chart
            charts['risk_contribution'] = self.generate_risk_contribution_chart(
                list(market_data.columns), weights, performance_metrics
            )
            
            return charts
            
        except Exception as e:
            logger.error(f"Error generating chart data: {e}")
            return {}
    
    def generate_portfolio_performance_chart(self, market_data: pd.DataFrame, 
                                           weights: List[float]) -> Dict:
        """
        Generate portfolio performance line chart data
        """
        try:
            # Calculate portfolio returns
            returns = market_data.pct_change().dropna()

            # Ensure weights match the number of columns
            if len(weights) != len(returns.columns):
                logger.warning(f"Chart: Weights length {len(weights)} != columns {len(returns.columns)}")
                if len(weights) > len(returns.columns):
                    weights = weights[:len(returns.columns)]
                else:
                    weights = weights + [0.0] * (len(returns.columns) - len(weights))

            # Convert weights to numpy array for proper broadcasting
            weights_array = np.array(weights)
            portfolio_returns = (returns * weights_array).sum(axis=1)
            cumulative_returns = (1 + portfolio_returns).cumprod()
            
            # Prepare data for Chart.js
            dates = [date.strftime('%Y-%m-%d') for date in cumulative_returns.index]
            values = [float(val) for val in cumulative_returns.values]
            
            return {
                'type': 'line',
                'data': {
                    'labels': dates,
                    'datasets': [{
                        'label': 'Portfolio Performance',
                        'data': values,
                        'borderColor': self.color_palette[0],
                        'backgroundColor': self.color_palette[0] + '20',
                        'fill': True,
                        'tension': 0.1
                    }]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Portfolio Cumulative Performance'
                        },
                        'legend': {
                            'display': False
                        }
                    },
                    'scales': {
                        'y': {
                            'title': {
                                'display': True,
                                'text': 'Cumulative Return'
                            },
                            'ticks': {
                                'callback': 'function(value) { return (value - 1) * 100 + "%"; }'
                            }
                        },
                        'x': {
                            'title': {
                                'display': True,
                                'text': 'Date'
                            }
                        }
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating portfolio performance chart: {e}")
            return {}
    
    def generate_asset_allocation_chart(self, symbols: List[str], weights: List[float]) -> Dict:
        """
        Generate asset allocation pie chart data
        """
        try:
            return {
                'type': 'pie',
                'data': {
                    'labels': symbols,
                    'datasets': [{
                        'data': [float(w * 100) for w in weights],
                        'backgroundColor': self.color_palette[:len(symbols)],
                        'borderWidth': 2,
                        'borderColor': '#ffffff'
                    }]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Portfolio Asset Allocation'
                        },
                        'legend': {
                            'position': 'right'
                        },
                        'tooltip': {
                            'callbacks': {
                                'label': 'function(context) { return context.label + ": " + context.parsed + "%"; }'
                            }
                        }
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating asset allocation chart: {e}")
            return {}
    
    def generate_risk_radar_chart(self, risk_metrics: Dict) -> Dict:
        """
        Generate risk metrics radar chart
        """
        try:
            # Normalize metrics to 0-100 scale for radar chart
            metrics = {
                'Sharpe Ratio': min(100, max(0, (risk_metrics.get('sharpe_ratio', 0) + 2) * 25)),
                'Volatility': min(100, risk_metrics.get('annualized_volatility', 0) * 500),
                'Max Drawdown': min(100, abs(risk_metrics.get('max_drawdown', 0)) * 500),
                'VaR (95%)': min(100, abs(risk_metrics.get('var_95', 0)) * 1000),
                'Sortino Ratio': min(100, max(0, (risk_metrics.get('sortino_ratio', 0) + 2) * 25)),
                'Beta': min(100, max(0, risk_metrics.get('beta', 1) * 50))
            }
            
            return {
                'type': 'radar',
                'data': {
                    'labels': list(metrics.keys()),
                    'datasets': [{
                        'label': 'Risk Profile',
                        'data': list(metrics.values()),
                        'borderColor': self.color_palette[2],
                        'backgroundColor': self.color_palette[2] + '30',
                        'pointBackgroundColor': self.color_palette[2],
                        'pointBorderColor': '#ffffff',
                        'pointHoverBackgroundColor': '#ffffff',
                        'pointHoverBorderColor': self.color_palette[2]
                    }]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Risk Profile Analysis'
                        }
                    },
                    'scales': {
                        'r': {
                            'beginAtZero': True,
                            'max': 100,
                            'ticks': {
                                'stepSize': 20
                            }
                        }
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating risk radar chart: {e}")
            return {}
    
    def generate_correlation_heatmap(self, market_data: pd.DataFrame) -> Dict:
        """
        Generate correlation heatmap data
        """
        try:
            returns = market_data.pct_change().dropna()
            correlation_matrix = returns.corr()
            
            symbols = list(correlation_matrix.columns)
            
            # Prepare data for heatmap
            heatmap_data = []
            for i, row_symbol in enumerate(symbols):
                for j, col_symbol in enumerate(symbols):
                    heatmap_data.append({
                        'x': j,
                        'y': i,
                        'v': float(correlation_matrix.iloc[i, j]),
                        'row_label': row_symbol,
                        'col_label': col_symbol
                    })
            
            return {
                'type': 'heatmap',
                'data': heatmap_data,
                'labels': symbols,
                'options': {
                    'responsive': True,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Asset Correlation Matrix'
                        },
                        'tooltip': {
                            'callbacks': {
                                'title': 'function(context) { return context[0].raw.row_label + " vs " + context[0].raw.col_label; }',
                                'label': 'function(context) { return "Correlation: " + context.raw.v.toFixed(3); }'
                            }
                        }
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating correlation heatmap: {e}")
            return {}
    
    def generate_individual_performance_chart(self, market_data: pd.DataFrame, 
                                            performance_metrics: Dict) -> Dict:
        """
        Generate individual asset performance comparison
        """
        try:
            symbols = list(market_data.columns)
            individual_perf = performance_metrics.get('individual_performance', {})
            
            returns_data = []
            volatility_data = []
            
            for symbol in symbols:
                if symbol in individual_perf:
                    returns_data.append(individual_perf[symbol]['total_return'] * 100)
                    volatility_data.append(individual_perf[symbol]['volatility'] * 100)
                else:
                    returns_data.append(0)
                    volatility_data.append(0)
            
            return {
                'type': 'bar',
                'data': {
                    'labels': symbols,
                    'datasets': [
                        {
                            'label': 'Total Return (%)',
                            'data': returns_data,
                            'backgroundColor': self.color_palette[0],
                            'yAxisID': 'y'
                        },
                        {
                            'label': 'Volatility (%)',
                            'data': volatility_data,
                            'backgroundColor': self.color_palette[1],
                            'yAxisID': 'y1',
                            'type': 'line'
                        }
                    ]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Individual Asset Performance'
                        }
                    },
                    'scales': {
                        'y': {
                            'type': 'linear',
                            'display': True,
                            'position': 'left',
                            'title': {
                                'display': True,
                                'text': 'Return (%)'
                            }
                        },
                        'y1': {
                            'type': 'linear',
                            'display': True,
                            'position': 'right',
                            'title': {
                                'display': True,
                                'text': 'Volatility (%)'
                            },
                            'grid': {
                                'drawOnChartArea': False
                            }
                        }
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating individual performance chart: {e}")
            return {}
    
    def generate_volatility_comparison_chart(self, market_data: pd.DataFrame, 
                                           performance_metrics: Dict) -> Dict:
        """
        Generate volatility comparison chart
        """
        try:
            symbols = list(market_data.columns)
            individual_perf = performance_metrics.get('individual_performance', {})
            
            volatility_data = []
            for symbol in symbols:
                if symbol in individual_perf:
                    volatility_data.append(individual_perf[symbol]['volatility'] * 100)
                else:
                    volatility_data.append(0)
            
            return {
                'type': 'doughnut',
                'data': {
                    'labels': symbols,
                    'datasets': [{
                        'data': volatility_data,
                        'backgroundColor': self.color_palette[:len(symbols)],
                        'borderWidth': 2,
                        'borderColor': '#ffffff'
                    }]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Volatility Comparison'
                        },
                        'tooltip': {
                            'callbacks': {
                                'label': 'function(context) { return context.label + ": " + context.parsed.toFixed(2) + "%"; }'
                            }
                        }
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating volatility comparison chart: {e}")
            return {}
    
    def generate_returns_distribution_chart(self, market_data: pd.DataFrame, 
                                          weights: List[float]) -> Dict:
        """
        Generate returns distribution histogram
        """
        try:
            returns = market_data.pct_change().dropna()

            # Ensure weights match the number of columns
            if len(weights) != len(returns.columns):
                logger.warning(f"Returns chart: Weights length {len(weights)} != columns {len(returns.columns)}")
                if len(weights) > len(returns.columns):
                    weights = weights[:len(returns.columns)]
                else:
                    weights = weights + [0.0] * (len(returns.columns) - len(weights))

            # Convert weights to numpy array for proper broadcasting
            weights_array = np.array(weights)
            portfolio_returns = (returns * weights_array).sum(axis=1)
            
            # Create histogram bins
            hist, bin_edges = np.histogram(portfolio_returns * 100, bins=30)
            bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
            
            return {
                'type': 'bar',
                'data': {
                    'labels': [f"{x:.2f}%" for x in bin_centers],
                    'datasets': [{
                        'label': 'Frequency',
                        'data': hist.tolist(),
                        'backgroundColor': self.color_palette[3] + '80',
                        'borderColor': self.color_palette[3],
                        'borderWidth': 1
                    }]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Daily Returns Distribution'
                        },
                        'legend': {
                            'display': False
                        }
                    },
                    'scales': {
                        'x': {
                            'title': {
                                'display': True,
                                'text': 'Daily Return (%)'
                            }
                        },
                        'y': {
                            'title': {
                                'display': True,
                                'text': 'Frequency'
                            }
                        }
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating returns distribution chart: {e}")
            return {}
    
    def generate_risk_contribution_chart(self, symbols: List[str], weights: List[float], 
                                       performance_metrics: Dict) -> Dict:
        """
        Generate risk contribution chart
        """
        try:
            individual_perf = performance_metrics.get('individual_performance', {})
            
            risk_contributions = []
            for symbol in symbols:
                if symbol in individual_perf:
                    # Simplified risk contribution calculation
                    weight = individual_perf[symbol]['weight']
                    volatility = individual_perf[symbol]['volatility']
                    risk_contrib = weight * volatility
                    risk_contributions.append(risk_contrib * 100)
                else:
                    risk_contributions.append(0)
            
            return {
                'type': 'horizontalBar',
                'data': {
                    'labels': symbols,
                    'datasets': [{
                        'label': 'Risk Contribution (%)',
                        'data': risk_contributions,
                        'backgroundColor': self.color_palette[:len(symbols)],
                        'borderWidth': 1
                    }]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Risk Contribution by Asset'
                        }
                    },
                    'scales': {
                        'x': {
                            'title': {
                                'display': True,
                                'text': 'Risk Contribution (%)'
                            }
                        }
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating risk contribution chart: {e}")
            return {}
