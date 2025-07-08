"""
Utility Functions - Helper functions for data validation, formatting, and common operations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def validate_portfolio_input(data: Dict) -> Dict[str, Any]:
    """
    Validate portfolio input data
    
    Args:
        data: Dictionary containing 'stocks' and 'weights'
    
    Returns:
        Dictionary with 'valid' boolean and 'message' string
    """
    try:
        if not isinstance(data, dict):
            return {'valid': False, 'message': 'Input must be a dictionary'}
        
        if 'stocks' not in data or 'weights' not in data:
            return {'valid': False, 'message': 'Missing required fields: stocks and weights'}
        
        stocks = data['stocks']
        weights = data['weights']
        
        # Validate stocks
        if not isinstance(stocks, list) or len(stocks) == 0:
            return {'valid': False, 'message': 'Stocks must be a non-empty list'}
        
        if len(stocks) > 20:
            return {'valid': False, 'message': 'Maximum 20 stocks allowed in portfolio'}
        
        # Validate stock symbols
        for stock in stocks:
            if not isinstance(stock, str) or not is_valid_stock_symbol(stock):
                return {'valid': False, 'message': f'Invalid stock symbol: {stock}'}
        
        # Check for duplicates
        if len(stocks) != len(set(stocks)):
            return {'valid': False, 'message': 'Duplicate stock symbols found'}
        
        # Validate weights
        if not isinstance(weights, list) or len(weights) == 0:
            return {'valid': False, 'message': 'Weights must be a non-empty list'}
        
        if len(stocks) != len(weights):
            return {'valid': False, 'message': 'Number of stocks and weights must match'}
        
        # Validate weight values
        for i, weight in enumerate(weights):
            if not isinstance(weight, (int, float)):
                return {'valid': False, 'message': f'Weight {i+1} must be a number'}
            
            if weight < 0:
                return {'valid': False, 'message': f'Weight {i+1} cannot be negative'}
            
            if weight > 1:
                return {'valid': False, 'message': f'Weight {i+1} cannot exceed 100%'}
        
        # Check if weights sum to approximately 1
        total_weight = sum(weights)
        if abs(total_weight - 1.0) > 0.01:  # Allow 1% tolerance
            return {'valid': False, 'message': f'Weights must sum to 100% (currently {total_weight*100:.1f}%)'}
        
        return {'valid': True, 'message': 'Input validation successful'}
        
    except Exception as e:
        logger.error(f"Error in input validation: {e}")
        return {'valid': False, 'message': f'Validation error: {str(e)}'}

def is_valid_stock_symbol(symbol: str) -> bool:
    """
    Check if a stock symbol is valid format
    
    Args:
        symbol: Stock symbol string
    
    Returns:
        Boolean indicating if symbol is valid
    """
    if not isinstance(symbol, str):
        return False
    
    # Basic validation: 1-5 characters, letters only, uppercase
    pattern = r'^[A-Z]{1,5}$'
    return bool(re.match(pattern, symbol.upper()))

def format_response(data: Dict) -> Dict:
    """
    Format API response with consistent structure
    
    Args:
        data: Raw response data
    
    Returns:
        Formatted response dictionary
    """
    try:
        formatted = {
            'success': True,
            'data': data,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'processing_time': None  # Could be calculated if needed
            }
        }
        
        return formatted
        
    except Exception as e:
        logger.error(f"Error formatting response: {e}")
        return {
            'success': False,
            'error': str(e),
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            }
        }

def format_currency(value: float, currency: str = 'USD') -> str:
    """
    Format currency values for display
    
    Args:
        value: Numeric value
        currency: Currency code (default: USD)
    
    Returns:
        Formatted currency string
    """
    try:
        if currency == 'USD':
            if abs(value) >= 1e9:
                return f"${value/1e9:.2f}B"
            elif abs(value) >= 1e6:
                return f"${value/1e6:.2f}M"
            elif abs(value) >= 1e3:
                return f"${value/1e3:.2f}K"
            else:
                return f"${value:.2f}"
        else:
            return f"{value:.2f} {currency}"
    except Exception:
        return str(value)

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Format percentage values for display
    
    Args:
        value: Decimal value (e.g., 0.15 for 15%)
        decimal_places: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    try:
        return f"{value * 100:.{decimal_places}f}%"
    except Exception:
        return str(value)

def calculate_date_range(period: str) -> Tuple[datetime, datetime]:
    """
    Calculate start and end dates for a given period
    
    Args:
        period: Period string ('1d', '1w', '1m', '3m', '6m', '1y', '2y', '5y')
    
    Returns:
        Tuple of (start_date, end_date)
    """
    end_date = datetime.now()
    
    period_mapping = {
        '1d': timedelta(days=1),
        '1w': timedelta(weeks=1),
        '1m': timedelta(days=30),
        '3m': timedelta(days=90),
        '6m': timedelta(days=180),
        '1y': timedelta(days=365),
        '2y': timedelta(days=730),
        '5y': timedelta(days=1825),
        '10y': timedelta(days=3650)
    }
    
    delta = period_mapping.get(period, timedelta(days=365))
    start_date = end_date - delta
    
    return start_date, end_date

def clean_numeric_data(data: Any) -> Any:
    """
    Clean numeric data by handling NaN, infinity, and converting to appropriate types
    
    Args:
        data: Input data (can be various types)
    
    Returns:
        Cleaned data
    """
    if isinstance(data, (list, tuple)):
        return [clean_numeric_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: clean_numeric_data(value) for key, value in data.items()}
    elif isinstance(data, pd.Series):
        return data.fillna(0).replace([np.inf, -np.inf], 0)
    elif isinstance(data, pd.DataFrame):
        return data.fillna(0).replace([np.inf, -np.inf], 0)
    elif isinstance(data, (int, float)):
        if np.isnan(data) or np.isinf(data):
            return 0
        return data
    else:
        return data

def calculate_business_days(start_date: datetime, end_date: datetime) -> int:
    """
    Calculate number of business days between two dates
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        Number of business days
    """
    return pd.bdate_range(start_date, end_date).size

def generate_color_palette(n_colors: int) -> List[str]:
    """
    Generate a color palette for charts
    
    Args:
        n_colors: Number of colors needed
    
    Returns:
        List of hex color codes
    """
    # Professional color palette
    base_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf'   # Cyan
    ]
    
    if n_colors <= len(base_colors):
        return base_colors[:n_colors]
    
    # Generate additional colors if needed
    import colorsys
    colors = base_colors.copy()
    
    for i in range(len(base_colors), n_colors):
        hue = (i * 0.618033988749895) % 1  # Golden ratio for good distribution
        saturation = 0.7
        value = 0.8
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    
    return colors

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
    
    Returns:
        Division result or default value
    """
    try:
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return default
        result = numerator / denominator
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except Exception:
        return default

def round_to_significant_figures(value: float, sig_figs: int = 4) -> float:
    """
    Round a number to specified significant figures
    
    Args:
        value: Input value
        sig_figs: Number of significant figures
    
    Returns:
        Rounded value
    """
    try:
        if value == 0:
            return 0
        return round(value, -int(np.floor(np.log10(abs(value)))) + (sig_figs - 1))
    except Exception:
        return value

def log_performance(func_name: str, start_time: datetime, end_time: datetime, **kwargs):
    """
    Log performance metrics for functions
    
    Args:
        func_name: Name of the function
        start_time: Function start time
        end_time: Function end time
        **kwargs: Additional metadata to log
    """
    duration = (end_time - start_time).total_seconds()
    
    log_data = {
        'function': func_name,
        'duration_seconds': duration,
        'timestamp': end_time.isoformat(),
        **kwargs
    }
    
    logger.info(f"Performance: {func_name} completed in {duration:.3f}s", extra=log_data)
