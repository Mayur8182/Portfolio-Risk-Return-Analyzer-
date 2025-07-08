# Portfolio Risk Analysis Dashboard - API Documentation

## 📡 API Overview

The Portfolio Risk Analysis Dashboard provides a RESTful API for portfolio analysis, optimization, and sentiment analysis. All endpoints return JSON responses and support CORS for cross-origin requests.

**Base URL**: `http://localhost:5000/api`

## 🔐 Authentication

Currently, no authentication is required for API access. This is suitable for development and demo purposes.

## 📊 API Endpoints

### 1. Portfolio Analysis

**Endpoint**: `POST /api/analyze`

**Description**: Analyzes a portfolio and returns comprehensive risk metrics, performance data, and visualizations.

#### Request Body

```json
{
  "stocks": ["AAPL", "GOOGL", "MSFT"],
  "weights": [0.4, 0.35, 0.25],
  "period": "1y",
  "risk_free_rate": 0.02
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `stocks` | array | Yes | Array of stock symbols (uppercase) |
| `weights` | array | Yes | Array of portfolio weights (must sum to 1.0) |
| `period` | string | No | Analysis period: "1d", "1w", "1m", "3m", "6m", "1y", "2y", "5y" (default: "1y") |
| `risk_free_rate` | number | No | Annual risk-free rate as decimal (default: 0.02) |

#### Response

```json
{
  "success": true,
  "data": {
    "portfolio": {
      "stocks": ["AAPL", "GOOGL", "MSFT"],
      "weights": [0.4, 0.35, 0.25],
      "total_value": 1.0
    },
    "risk_metrics": {
      "daily_return_mean": 0.001234,
      "daily_return_std": 0.0156,
      "annualized_return": 0.1234,
      "annualized_volatility": 0.1876,
      "sharpe_ratio": 1.234,
      "sortino_ratio": 1.456,
      "calmar_ratio": 0.789,
      "var_95": -0.0234,
      "var_99": -0.0345,
      "cvar_95": -0.0267,
      "cvar_99": -0.0389,
      "max_drawdown": -0.1234,
      "beta": 1.05,
      "skewness": -0.123,
      "kurtosis": 2.345,
      "total_return": 0.1567,
      "volatility_rank": "Moderate"
    },
    "performance_metrics": {
      "portfolio_total_return": 0.1567,
      "portfolio_annualized_return": 0.1234,
      "individual_performance": {
        "AAPL": {
          "weight": 0.4,
          "total_return": 0.1789,
          "annualized_return": 0.1456,
          "volatility": 0.2123,
          "sharpe_ratio": 1.234,
          "contribution_to_return": 0.0584,
          "contribution_to_risk": 0.0181
        }
      },
      "attribution": {
        "total_return_contribution": 0.1234,
        "total_risk_contribution": 0.0352
      },
      "best_performer": "AAPL",
      "worst_performer": "MSFT",
      "most_volatile": "GOOGL",
      "least_volatile": "MSFT"
    },
    "correlation_matrix": {
      "matrix": {
        "AAPL": {"AAPL": 1.0, "GOOGL": 0.67, "MSFT": 0.78},
        "GOOGL": {"AAPL": 0.67, "GOOGL": 1.0, "MSFT": 0.72},
        "MSFT": {"AAPL": 0.78, "GOOGL": 0.72, "MSFT": 1.0}
      },
      "statistics": {
        "average_correlation": 0.723,
        "max_correlation": 0.78,
        "min_correlation": 0.67,
        "most_correlated_pair": ["AAPL", "MSFT"],
        "least_correlated_pair": ["AAPL", "GOOGL"]
      },
      "diversification_ratio": 1.234
    },
    "optimization": {
      "max_sharpe": {
        "weights": [0.45, 0.30, 0.25],
        "metrics": {
          "expected_return": 0.1345,
          "volatility": 0.1678,
          "sharpe_ratio": 1.456
        },
        "success": true
      }
    },
    "sentiment": {
      "individual_sentiment": {
        "AAPL": {
          "symbol": "AAPL",
          "overall_score": 0.234,
          "sentiment_label": "Positive",
          "components": {
            "news": {"score": 0.3, "confidence": 0.8},
            "social": {"score": 0.2, "confidence": 0.6},
            "technical": {"score": 0.1, "confidence": 0.7}
          },
          "confidence": 0.7
        }
      },
      "portfolio_sentiment": {
        "overall_score": 0.189,
        "sentiment_label": "Positive",
        "confidence": 0.72
      }
    },
    "charts": {
      "portfolio_performance": {
        "type": "line",
        "data": {
          "labels": ["2023-01-01", "2023-01-02", "..."],
          "datasets": [...]
        }
      }
    },
    "timestamp": "2023-12-01T10:30:00Z"
  },
  "metadata": {
    "timestamp": "2023-12-01T10:30:00Z",
    "version": "1.0.0"
  }
}
```

#### Error Response

```json
{
  "success": false,
  "error": "Invalid stock symbol: INVALID",
  "metadata": {
    "timestamp": "2023-12-01T10:30:00Z",
    "version": "1.0.0"
  }
}
```

### 2. Portfolio Optimization

**Endpoint**: `POST /api/optimize`

**Description**: Optimizes portfolio weights based on specified objective.

#### Request Body

```json
{
  "stocks": ["AAPL", "GOOGL", "MSFT"],
  "objective": "max_sharpe"
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `stocks` | array | Yes | Array of stock symbols |
| `objective` | string | No | Optimization objective: "max_sharpe", "min_risk", "max_return" (default: "max_sharpe") |

#### Response

```json
{
  "max_sharpe": {
    "weights": [0.45, 0.30, 0.25],
    "metrics": {
      "expected_return": 0.1345,
      "volatility": 0.1678,
      "sharpe_ratio": 1.456,
      "variance": 0.0281
    },
    "success": true,
    "message": "Optimization successful"
  },
  "equal_weight_benchmark": {
    "weights": [0.333, 0.333, 0.334],
    "metrics": {
      "expected_return": 0.1234,
      "volatility": 0.1789,
      "sharpe_ratio": 1.234
    },
    "symbols": ["AAPL", "GOOGL", "MSFT"]
  }
}
```

### 3. Sentiment Analysis

**Endpoint**: `GET /api/sentiment/{symbol}`

**Description**: Get sentiment analysis for a specific stock symbol.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbol` | string | Yes | Stock symbol (in URL path) |

#### Response

```json
{
  "symbol": "AAPL",
  "overall_score": 0.234,
  "sentiment_label": "Positive",
  "components": {
    "news": {
      "score": 0.3,
      "articles_analyzed": 15,
      "source": "financial_news",
      "confidence": 0.8,
      "last_updated": "2023-12-01T10:30:00Z"
    },
    "social": {
      "score": 0.2,
      "mentions_analyzed": 250,
      "source": "social_media",
      "confidence": 0.6,
      "last_updated": "2023-12-01T10:30:00Z"
    },
    "technical": {
      "score": 0.1,
      "indicators": {
        "rsi": 65.4,
        "macd_signal": 1,
        "trend": "bullish"
      },
      "source": "technical_analysis",
      "confidence": 0.7,
      "last_updated": "2023-12-01T10:30:00Z"
    }
  },
  "analysis_timestamp": "2023-12-01T10:30:00Z",
  "confidence": 0.7
}
```

### 4. Market Data

**Endpoint**: `GET /api/market-data/{symbol}`

**Description**: Get market data for a specific stock symbol.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbol` | string | Yes | Stock symbol (in URL path) |
| `period` | string | No | Time period (query parameter) |

#### Example Request

```
GET /api/market-data/AAPL?period=1y
```

#### Response

```json
{
  "symbol": "AAPL",
  "prices": {
    "2023-01-01": {
      "Open": 150.0,
      "High": 152.5,
      "Low": 149.0,
      "Close": 151.2,
      "Adj Close": 151.2,
      "Volume": 50000000
    }
  },
  "returns": {
    "2023-01-02": 0.0123,
    "2023-01-03": -0.0056
  },
  "summary": {
    "current_price": 151.2,
    "previous_close": 150.0,
    "change": 1.2,
    "change_percent": 0.8,
    "volume": 50000000,
    "high_52w": 180.0,
    "low_52w": 120.0,
    "avg_volume": 45000000,
    "volatility": 0.25
  },
  "metadata": {
    "start_date": "2023-01-01",
    "end_date": "2023-12-01",
    "data_points": 252,
    "last_updated": "2023-12-01T10:30:00Z"
  }
}
```

### 5. Health Check

**Endpoint**: `GET /api/health`

**Description**: Check API health status.

#### Response

```json
{
  "status": "healthy",
  "timestamp": "2023-12-01T10:30:00Z",
  "version": "1.0.0"
}
```

## 🚨 Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 404 | Not Found (invalid endpoint) |
| 500 | Internal Server Error |

### Error Response Format

```json
{
  "success": false,
  "error": "Error description",
  "metadata": {
    "timestamp": "2023-12-01T10:30:00Z",
    "version": "1.0.0"
  }
}
```

### Common Error Messages

- `"Missing required fields: stocks and weights"`
- `"Invalid stock symbol: {symbol}"`
- `"Weights must sum to 100%"`
- `"Failed to fetch market data"`
- `"Analysis failed: {reason}"`
- `"Optimization failed: {reason}"`

## 📝 Usage Examples

### Python Example

```python
import requests
import json

# Portfolio analysis
portfolio_data = {
    "stocks": ["AAPL", "GOOGL", "MSFT"],
    "weights": [0.4, 0.35, 0.25]
}

response = requests.post(
    "http://localhost:5000/api/analyze",
    headers={"Content-Type": "application/json"},
    data=json.dumps(portfolio_data)
)

if response.status_code == 200:
    result = response.json()
    print(f"Sharpe Ratio: {result['data']['risk_metrics']['sharpe_ratio']}")
else:
    print(f"Error: {response.json()['error']}")
```

### JavaScript Example

```javascript
// Portfolio analysis
const portfolioData = {
    stocks: ['AAPL', 'GOOGL', 'MSFT'],
    weights: [0.4, 0.35, 0.25]
};

fetch('http://localhost:5000/api/analyze', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(portfolioData)
})
.then(response => response.json())
.then(data => {
    if (data.success) {
        console.log('Sharpe Ratio:', data.data.risk_metrics.sharpe_ratio);
    } else {
        console.error('Error:', data.error);
    }
})
.catch(error => console.error('Network error:', error));
```

### cURL Example

```bash
# Portfolio analysis
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "stocks": ["AAPL", "GOOGL", "MSFT"],
    "weights": [0.4, 0.35, 0.25]
  }'

# Get sentiment for a stock
curl http://localhost:5000/api/sentiment/AAPL

# Health check
curl http://localhost:5000/api/health
```

## 🔧 Rate Limiting

Currently, no rate limiting is implemented. For production use, consider implementing rate limiting to prevent abuse.

## 📊 Data Sources

- **Market Data**: Yahoo Finance (via yfinance library)
- **News Sentiment**: Simulated (replace with actual news API)
- **Social Sentiment**: Simulated (replace with Twitter/Reddit APIs)
- **Technical Indicators**: Calculated from price data

## 🔒 Security Considerations

- **Input Validation**: All inputs are validated
- **CORS**: Enabled for cross-origin requests
- **No Authentication**: Currently no auth required
- **Data Sanitization**: Stock symbols are sanitized and validated

For production deployment, consider adding:
- API key authentication
- Rate limiting
- Input sanitization
- HTTPS enforcement
- Request logging
