# Portfolio Risk Analysis Dashboard - Setup Guide

## 🚀 Quick Start

This guide will help you set up and run the Portfolio Risk Analysis Dashboard on your local machine.

## 📋 Prerequisites

- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **pip** (Python package installer)
- **Modern web browser** (Chrome, Firefox, Safari, Edge)
- **Internet connection** (for fetching market data)

## 🛠️ Installation Steps

### 1. Clone or Download the Project

If you have the project files, ensure your directory structure looks like this:

```
portfolio-dashboard/
├── backend/
├── frontend/
├── static/
├── visualizations/
├── data/
├── tests/
├── requirements.txt
└── README.md
```

### 2. Set Up Python Environment

#### Option A: Using Virtual Environment (Recommended)

```bash
# Navigate to project directory
cd portfolio-dashboard

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Option B: Using Conda

```bash
# Create conda environment
conda create -n portfolio-dashboard python=3.9
conda activate portfolio-dashboard
```

### 3. Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Note**: If you encounter any installation errors, try:

```bash
# Upgrade pip first
pip install --upgrade pip

# Install packages one by one if needed
pip install Flask==2.3.3
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install yfinance==0.2.18
# ... continue with other packages from requirements.txt
```

### 4. Verify Installation

Run the test suite to ensure everything is working:

```bash
# Navigate to the project root
cd portfolio-dashboard

# Run tests
python tests/test_analyze.py
```

You should see output indicating that tests are running and passing.

## 🏃‍♂️ Running the Application

### 1. Start the Backend Server

```bash
# Navigate to backend directory
cd backend

# Start the Flask server
python app.py
```

You should see output similar to:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### 2. Access the Frontend

Open your web browser and navigate to:

```
http://localhost:5000
```

You should see the Portfolio Risk Analysis Dashboard landing page.

## 🧪 Testing the Application

### 1. Basic Functionality Test

1. **Open the application** in your browser
2. **Enter a test portfolio**:
   - Stocks: AAPL, GOOGL, MSFT
   - Weights: 40%, 35%, 25%
3. **Click "Analyze Portfolio"**
4. **Wait for analysis** (may take 30-60 seconds for first run)
5. **View the dashboard** with charts and metrics

### 2. Using Preset Portfolios

Try the preset portfolios available on the main page:
- **Tech Portfolio**: AAPL, GOOGL, MSFT
- **Diversified**: SPY, QQQ, VTI, IWM
- **Growth**: TSLA, NVDA, AMD, NFLX

### 3. Testing Different Features

- **Portfolio Optimization**: Click "Optimize Portfolio" in the dashboard
- **Data Refresh**: Click "Refresh Data" to update market data
- **Export Report**: Click "Export Report" to download analysis

## 🔧 Configuration Options

### 1. Risk-Free Rate

You can adjust the risk-free rate used in calculations:
- Default: 2.0% annually
- Modify in the frontend form or backend configuration

### 2. Analysis Period

Choose different time periods for historical analysis:
- 1 Year (default)
- 2 Years
- 3 Years
- 5 Years

### 3. API Configuration

The application uses Yahoo Finance for market data. No API key required for basic functionality.

For production use, consider:
- **Alpha Vantage API**: For more reliable data
- **News API**: For sentiment analysis
- **Twitter API**: For social sentiment

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. "Module not found" errors

```bash
# Ensure you're in the correct directory and virtual environment is activated
pip install -r requirements.txt
```

#### 2. "Port already in use" error

```bash
# Kill any existing Flask processes
# On Windows:
taskkill /f /im python.exe
# On macOS/Linux:
pkill -f python
```

Or change the port in `backend/app.py`:
```python
app.run(host='0.0.0.0', port=5001, debug=True)  # Use port 5001 instead
```

#### 3. Data fetching errors

- **Check internet connection**
- **Verify stock symbols** are valid (use uppercase)
- **Try with fewer stocks** if timeout occurs
- **Wait and retry** if Yahoo Finance is temporarily unavailable

#### 4. Charts not displaying

- **Check browser console** for JavaScript errors
- **Ensure Chart.js is loading** (check network tab)
- **Try refreshing the page**
- **Clear browser cache**

#### 5. Slow performance

- **Reduce analysis period** (use 1 year instead of 5 years)
- **Use fewer stocks** in portfolio (max 5-6 for best performance)
- **Check system resources** (RAM, CPU usage)

### Debug Mode

To enable detailed error logging:

1. **Backend debugging**:
   ```python
   # In backend/app.py, ensure debug=True
   app.run(debug=True)
   ```

2. **Frontend debugging**:
   - Open browser Developer Tools (F12)
   - Check Console tab for JavaScript errors
   - Check Network tab for API call failures

## 📊 Sample Portfolios for Testing

### Conservative Portfolio
- **Stocks**: SPY, BND, VTI
- **Weights**: 60%, 30%, 10%

### Aggressive Growth
- **Stocks**: QQQ, ARKK, TSLA
- **Weights**: 50%, 30%, 20%

### Balanced Portfolio
- **Stocks**: SPY, QQQ, IWM, VEA, BND
- **Weights**: 40%, 20%, 15%, 15%, 10%

## 🔒 Security Notes

- This is a **development/demo application**
- **Do not use in production** without proper security measures
- **No sensitive data** should be entered
- **API keys** should be stored in environment variables for production

## 📈 Performance Optimization

For better performance:

1. **Enable caching** (already implemented for 1 hour)
2. **Use SSD storage** for faster data access
3. **Increase RAM** for larger portfolios
4. **Use production WSGI server** (Gunicorn, uWSGI) instead of Flask dev server

## 🆘 Getting Help

If you encounter issues:

1. **Check this troubleshooting guide**
2. **Review error messages** carefully
3. **Check browser console** for frontend issues
4. **Verify all dependencies** are installed correctly
5. **Try with a simple portfolio** first (2-3 stocks)

## 🎯 Next Steps

Once you have the basic application running:

1. **Explore different portfolios** and analysis periods
2. **Try portfolio optimization** features
3. **Export and review** analysis reports
4. **Customize the code** for your specific needs
5. **Add new features** like additional risk metrics or data sources

## 📝 Development Notes

- **Backend**: Flask-based REST API
- **Frontend**: Vanilla JavaScript with Chart.js
- **Data Source**: Yahoo Finance (yfinance library)
- **Styling**: Bootstrap 5 + Tailwind CSS
- **Charts**: Chart.js for interactive visualizations

The application is designed to be educational and demonstrate financial analysis concepts. For production use, additional features like user authentication, data persistence, and enhanced error handling would be needed.
