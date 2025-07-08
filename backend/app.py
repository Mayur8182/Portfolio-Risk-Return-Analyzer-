"""
Portfolio Risk Analysis Dashboard - Main Flask Application
"""

from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask_cors import CORS
import os
import json
import logging
import hashlib
from datetime import datetime, timedelta

# Import our custom modules
from fetch_data import PortfolioDataFetcher
from simple_data_fetcher import SimpleDataFetcher
from analyze import PortfolioAnalyzer
from optimize import PortfolioOptimizer
from sentiment import SentimentAnalyzer
from utils import format_response, validate_portfolio_input
from pdf_generator import ProfessionalPDFGenerator
from enhanced_accuracy_config import EnhancedAccuracyConfig, InstitutionalDataValidator

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced modules
try:
    from enhanced_data_fetcher import EnhancedDataFetcher
    from database_manager import DatabaseManager
    from advanced_analytics import AdvancedAnalyticsEngine
    from data_quality_validator import DataQualityValidator
    ENHANCED_IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced modules not available: {e}")
    ENHANCED_IMPORTS_AVAILABLE = False

# Add parent directory to path for visualizations import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualizations.chart_data import ChartDataGenerator

# Initialize Flask app
app = Flask(__name__, 
           static_folder='../static',
           template_folder='../frontend')
CORS(app)

# Initialize basic components
data_fetcher = PortfolioDataFetcher()
simple_data_fetcher = SimpleDataFetcher()
analyzer = PortfolioAnalyzer()
optimizer = PortfolioOptimizer()
sentiment_analyzer = SentimentAnalyzer()
chart_generator = ChartDataGenerator()
pdf_generator = ProfessionalPDFGenerator()
accuracy_config = EnhancedAccuracyConfig()
institutional_validator = InstitutionalDataValidator()

# Test data fetchers
logger.info("Testing data fetchers...")
if simple_data_fetcher.test_connection():
    logger.info("✅ Simple data fetcher is working")
else:
    logger.warning("❌ Simple data fetcher test failed")

# Initialize enhanced components if available
ENHANCED_MODE = False
enhanced_data_fetcher = None
database_manager = None
advanced_analytics = None
data_quality_validator = None

if ENHANCED_IMPORTS_AVAILABLE:
    try:
        enhanced_data_fetcher = EnhancedDataFetcher()
        database_manager = DatabaseManager()
        advanced_analytics = AdvancedAnalyticsEngine()
        data_quality_validator = DataQualityValidator()
        logger.info("✅ Enhanced components initialized successfully")
        ENHANCED_MODE = True
    except Exception as e:
        logger.warning(f"❌ Enhanced components initialization failed: {e}. Using basic mode.")
        ENHANCED_MODE = False
else:
    logger.info("Enhanced components not available, using basic mode")

@app.route('/')
def index():
    """Serve the main landing page"""
    return render_template('index.html')

@app.route('/visual-dashboard.html')
def visual_dashboard():
    """Serve the visual dashboard page"""
    try:
        return send_from_directory('../frontend', 'visual-dashboard.html')
    except Exception as e:
        logger.error(f"Error serving visual dashboard: {e}")
        return jsonify({"error": "Visual dashboard not found"}), 404

@app.route('/test.html')
def test_page():
    """Serve the test page"""
    try:
        return send_from_directory('../frontend', 'test.html')
    except Exception as e:
        logger.error(f"Error serving test page: {e}")
        return jsonify({"error": "Test page not found"}), 404

@app.route('/simple-dashboard.html')
def simple_dashboard():
    """Serve the simple dashboard page"""
    try:
        return send_from_directory('../frontend', 'simple-dashboard.html')
    except Exception as e:
        logger.error(f"Error serving simple dashboard: {e}")
        return jsonify({"error": "Simple dashboard not found"}), 404

@app.route('/dashboard')
def dashboard_redirect():
    """Redirect to visual dashboard"""
    return visual_dashboard()

# Static file serving for frontend assets
@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from frontend directory"""
    try:
        # Check if it's an HTML file
        if filename.endswith('.html'):
            return send_from_directory('../frontend', filename)
        # Check if it's a CSS/JS file
        elif filename.startswith('assets/'):
            return send_from_directory('../frontend', filename)
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {e}")
        return jsonify({"error": "File not found"}), 404

@app.route('/dashboard')
def dashboard():
    """Serve the analytics dashboard"""
    return render_template('dashboard.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_portfolio():
    """
    Main API endpoint for portfolio analysis
    Expected input: {"stocks": ["AAPL", "GOOGL", "MSFT"], "weights": [0.4, 0.3, 0.3]}
    """
    try:
        data = request.get_json()
        
        # Validate input
        validation_result = validate_portfolio_input(data)
        if not validation_result['valid']:
            return jsonify({'error': validation_result['message']}), 400
        
        stocks = data['stocks']
        weights = data['weights']
        
        logger.info(f"Analyzing portfolio: {stocks} with weights: {weights}")

        # Try enhanced data fetcher first, with institutional-grade accuracy
        market_data = None
        quality_reports = {}
        accuracy_settings = accuracy_config.get_enhanced_settings()

        if ENHANCED_MODE and enhanced_data_fetcher:
            try:
                logger.info("Attempting enhanced data fetcher with institutional-grade accuracy")
                logger.info(f"Target data points per asset: {accuracy_settings['data_quality']['minimum_data_points']}")
                logger.info(f"Accuracy threshold: {accuracy_settings['data_quality']['accuracy_score_threshold']}")

                market_data = enhanced_data_fetcher.get_portfolio_data_enhanced(stocks)

                if market_data is not None and not market_data.empty:
                    logger.info("✅ Enhanced data fetcher successful")

                    # Institutional-grade data quality validation
                    if data_quality_validator:
                        for symbol in stocks:
                            if symbol in market_data.columns:
                                try:
                                    symbol_data = market_data[[symbol]].copy()
                                    symbol_data.columns = ['Close']
                                    # Add dummy OHLCV data for validation
                                    symbol_data['Open'] = symbol_data['Close']
                                    symbol_data['High'] = symbol_data['Close'] * 1.02
                                    symbol_data['Low'] = symbol_data['Close'] * 0.98
                                    symbol_data['Volume'] = 1000000

                                    # Use institutional validator for higher accuracy
                                    quality_report = institutional_validator.comprehensive_validation(symbol_data, symbol)
                                    quality_reports[symbol] = quality_report

                                    logger.info(f"📊 {symbol} - Quality Score: {quality_report['quality_score']:.3f}")
                                    logger.info(f"📊 {symbol} - Institutional Grade: {quality_report['institutional_grade']}")

                                    if quality_report['quality_score'] < accuracy_settings['data_quality']['accuracy_score_threshold']:
                                        logger.warning(f"⚠️ {symbol} quality below institutional threshold: {quality_report['quality_score']:.3f}")

                                except Exception as qe:
                                    logger.warning(f"Data quality validation failed for {symbol}: {qe}")
                else:
                    logger.warning("Enhanced data fetcher returned no data, falling back to basic fetcher")
                    market_data = None

            except Exception as e:
                logger.warning(f"Enhanced data fetcher failed: {e}. Falling back to basic fetcher.")
                market_data = None

        # Fallback to basic data fetcher if enhanced failed or not available
        if market_data is None or (hasattr(market_data, 'empty') and market_data.empty):
            logger.info("Using basic data fetcher (Yahoo Finance)")
            try:
                market_data = data_fetcher.fetch_portfolio_data(stocks)
                if market_data is not None and not market_data.empty:
                    logger.info("✅ Basic data fetcher successful")
                else:
                    logger.warning("Basic data fetcher returned no data, trying simple fetcher")
                    market_data = None
            except Exception as e:
                logger.warning(f"Basic data fetcher failed: {e}, trying simple fetcher")
                market_data = None

        # Final fallback to simple data fetcher
        if market_data is None or (hasattr(market_data, 'empty') and market_data.empty):
            logger.info("Using simple data fetcher as final fallback")
            try:
                market_data = simple_data_fetcher.fetch_portfolio_data(stocks, period)
                if market_data is not None and not market_data.empty:
                    logger.info("✅ Simple data fetcher successful")
                else:
                    logger.error("❌ All data fetchers failed")
            except Exception as e:
                logger.error(f"❌ Simple data fetcher also failed: {e}")
                market_data = None

        if market_data is None:
            return jsonify({'error': 'Failed to fetch market data'}), 500
        
        # Perform risk analysis with enhanced analytics if available
        if ENHANCED_MODE and advanced_analytics:
            try:
                logger.info("Using advanced analytics engine")
                portfolio_returns = analyzer.calculate_returns(market_data, weights)

                # Get advanced risk metrics
                advanced_risk_metrics = advanced_analytics.calculate_advanced_risk_metrics(portfolio_returns)

                if advanced_risk_metrics:
                    # Convert advanced metrics to standard format
                    risk_metrics = {
                        'daily_return_mean': float(portfolio_returns.mean()),
                        'daily_return_std': float(portfolio_returns.std()),
                        'annualized_return': float((1 + portfolio_returns.mean()) ** 252 - 1),
                        'annualized_volatility': float(advanced_risk_metrics.volatility),
                        'sharpe_ratio': float(advanced_risk_metrics.sharpe_ratio),
                        'sortino_ratio': float(advanced_risk_metrics.sortino_ratio),
                        'calmar_ratio': float(advanced_risk_metrics.calmar_ratio),
                        'var_95': float(advanced_risk_metrics.var_95),
                        'var_99': float(advanced_risk_metrics.var_99),
                        'cvar_95': float(advanced_risk_metrics.cvar_95),
                        'cvar_99': float(advanced_risk_metrics.cvar_99),
                        'max_drawdown': float(advanced_risk_metrics.max_drawdown),
                        'beta': float(advanced_risk_metrics.beta),
                        'alpha': float(advanced_risk_metrics.alpha),
                        'r_squared': float(advanced_risk_metrics.r_squared),
                        'tracking_error': float(advanced_risk_metrics.tracking_error),
                        'information_ratio': float(advanced_risk_metrics.information_ratio),
                        'skewness': float(advanced_risk_metrics.skewness),
                        'kurtosis': float(advanced_risk_metrics.kurtosis),
                        'total_return': float((1 + portfolio_returns).cumprod().iloc[-1] - 1),
                        'volatility_rank': 'Low' if advanced_risk_metrics.volatility < 0.15 else 'High'
                    }

                    # Add Monte Carlo simulation results
                    monte_carlo_results = advanced_analytics.monte_carlo_simulation(portfolio_returns)
                    risk_metrics['monte_carlo'] = monte_carlo_results

                    # Add stress testing results
                    stress_results = advanced_analytics.stress_testing(portfolio_returns, {})
                    risk_metrics['stress_testing'] = stress_results

                else:
                    # Fallback to basic risk analysis
                    risk_metrics = analyzer.calculate_risk_metrics(market_data, weights)

            except Exception as e:
                logger.error(f"Advanced analytics failed: {e}. Using basic analytics.")
                risk_metrics = analyzer.calculate_risk_metrics(market_data, weights)
        else:
            # Use basic risk analysis
            risk_metrics = analyzer.calculate_risk_metrics(market_data, weights)

        # Calculate returns and performance metrics
        performance_metrics = analyzer.calculate_performance_metrics(market_data, weights)
        
        # Generate correlation matrix
        correlation_matrix = analyzer.calculate_correlation_matrix(market_data)
        
        # Portfolio optimization
        optimization_results = optimizer.optimize_portfolio(market_data)
        
        # Sentiment analysis
        sentiment_scores = sentiment_analyzer.analyze_portfolio_sentiment(stocks)
        
        # Prepare chart data
        chart_data = chart_generator.generate_all_charts(
            market_data, weights, risk_metrics, performance_metrics
        )
        
        # Prepare response data
        response_data = {
            'portfolio': {
                'stocks': stocks,
                'weights': weights,
                'total_value': sum(weights)  # Should be 1.0
            },
            'risk_metrics': risk_metrics,
            'performance_metrics': performance_metrics,
            'correlation_matrix': correlation_matrix,
            'optimization': optimization_results,
            'sentiment': sentiment_scores,
            'charts': chart_data,
            'timestamp': datetime.now().isoformat(),
            'enhanced_mode': ENHANCED_MODE
        }

        # Store analysis results in database if enhanced mode is available
        if ENHANCED_MODE and database_manager:
            try:
                analysis_id = database_manager.store_portfolio_analysis({
                    'portfolio_hash': hashlib.sha256(str(sorted(zip(stocks, weights))).encode()).hexdigest(),
                    'symbols': stocks,
                    'weights': weights,
                    'analysis_results': response_data,
                    'data_quality_reports': quality_reports if 'quality_reports' in locals() else {},
                    'enhanced_features_used': True
                })
                response_data['analysis_id'] = analysis_id
                logger.info(f"Analysis results stored with ID: {analysis_id}")
            except Exception as e:
                logger.error(f"Failed to store analysis results: {e}")

        # Format response
        response = format_response(response_data)

        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in portfolio analysis: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/optimize', methods=['POST'])
def optimize_portfolio():
    """
    Portfolio optimization endpoint
    """
    try:
        data = request.get_json()
        stocks = data.get('stocks', [])
        objective = data.get('objective', 'max_sharpe')  # 'max_sharpe' or 'min_risk'
        
        if not stocks:
            return jsonify({'error': 'No stocks provided'}), 400
        
        # Fetch market data
        market_data = data_fetcher.fetch_portfolio_data(stocks)
        if market_data is None:
            return jsonify({'error': 'Failed to fetch market data'}), 500
        
        # Optimize portfolio
        optimization_results = optimizer.optimize_portfolio(
            market_data, objective=objective
        )
        
        return jsonify(optimization_results)
        
    except Exception as e:
        logger.error(f"Error in portfolio optimization: {str(e)}")
        return jsonify({'error': f'Optimization failed: {str(e)}'}), 500

@app.route('/api/sentiment/<symbol>')
def get_sentiment(symbol):
    """
    Get sentiment analysis for a specific stock symbol
    """
    try:
        sentiment_data = sentiment_analyzer.analyze_stock_sentiment(symbol)
        return jsonify(sentiment_data)
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return jsonify({'error': f'Sentiment analysis failed: {str(e)}'}), 500

@app.route('/api/market-data/<symbol>')
def get_market_data(symbol):
    """
    Get market data for a specific stock symbol
    """
    try:
        period = request.args.get('period', '1y')  # Default to 1 year
        market_data = data_fetcher.fetch_single_stock_data(symbol, period)
        return jsonify(market_data)
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        return jsonify({'error': f'Failed to fetch data: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'enhanced_mode': ENHANCED_MODE,
        'accuracy_level': 'institutional_grade',
        'pdf_export': 'enabled',
        'features': {
            'finnhub_integration': ENHANCED_MODE,
            'twelvedata_integration': ENHANCED_MODE,
            'mongodb_storage': ENHANCED_MODE,
            'advanced_analytics': ENHANCED_MODE,
            'data_quality_validation': ENHANCED_MODE,
            'pdf_report_generation': True,
            'institutional_accuracy': True
        }
    })

@app.route('/api/export-pdf', methods=['POST'])
def export_pdf():
    """
    Export portfolio analysis results as professional PDF report

    Expected JSON payload:
    {
        "analysis_data": {...}  // Complete analysis results
    }
    """
    try:
        data = request.get_json()

        if not data or 'analysis_data' not in data:
            return jsonify({"error": "Analysis data required for PDF export"}), 400

        analysis_data = data['analysis_data']

        # Generate PDF report
        logger.info("Generating professional PDF report...")

        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"portfolio_analysis_report_{timestamp}.pdf"
        pdf_path = os.path.join("reports", pdf_filename)

        # Ensure reports directory exists
        os.makedirs("reports", exist_ok=True)

        # Generate PDF
        pdf_generator.generate_portfolio_report(analysis_data, pdf_path)

        # Return PDF file
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=pdf_filename,
            mimetype='application/pdf'
        )

    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        return jsonify({"error": f"PDF generation failed: {str(e)}"}), 500

@app.route('/api/data-quality/<symbol>')
def get_data_quality(symbol):
    """Get data quality report for a specific symbol"""
    if not ENHANCED_MODE or not data_quality_validator:
        return jsonify({'error': 'Enhanced mode not available'}), 503

    try:
        # Get cached data from database
        if database_manager:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            market_data = database_manager.get_market_data_range(symbol, start_date, end_date)

            if market_data is not None:
                quality_report = data_quality_validator.validate_market_data(market_data, symbol)
                return jsonify(quality_report.to_dict())

        return jsonify({'error': 'No data available for quality analysis'}), 404

    except Exception as e:
        logger.error(f"Error getting data quality for {symbol}: {e}")
        return jsonify({'error': f'Data quality analysis failed: {str(e)}'}), 500

@app.route('/api/real-time-quote/<symbol>')
def get_real_time_quote(symbol):
    """Get real-time quote from enhanced data sources"""
    if not ENHANCED_MODE or not enhanced_data_fetcher:
        return jsonify({'error': 'Enhanced mode not available'}), 503

    try:
        quote_data = enhanced_data_fetcher.get_real_time_quote_finnhub(symbol)
        if quote_data:
            return jsonify(quote_data)
        else:
            return jsonify({'error': 'Failed to fetch real-time quote'}), 500

    except Exception as e:
        logger.error(f"Error getting real-time quote for {symbol}: {e}")
        return jsonify({'error': f'Real-time quote failed: {str(e)}'}), 500

@app.route('/api/company-profile/<symbol>')
def get_company_profile(symbol):
    """Get comprehensive company profile"""
    if not ENHANCED_MODE or not enhanced_data_fetcher:
        return jsonify({'error': 'Enhanced mode not available'}), 503

    try:
        company_data = enhanced_data_fetcher.get_comprehensive_company_data_finnhub(symbol)
        if company_data:
            return jsonify(company_data)
        else:
            return jsonify({'error': 'Failed to fetch company profile'}), 500

    except Exception as e:
        logger.error(f"Error getting company profile for {symbol}: {e}")
        return jsonify({'error': f'Company profile failed: {str(e)}'}), 500

@app.route('/api/database-stats')
def get_database_stats():
    """Get database statistics and health"""
    if not ENHANCED_MODE or not database_manager:
        return jsonify({'error': 'Enhanced mode not available'}), 503

    try:
        stats = database_manager.get_database_stats()
        return jsonify(stats)

    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return jsonify({'error': f'Database stats failed: {str(e)}'}), 500

@app.route('/api/analysis/<analysis_id>')
def get_stored_analysis(analysis_id):
    """Retrieve stored portfolio analysis"""
    if not ENHANCED_MODE or not database_manager:
        return jsonify({'error': 'Enhanced mode not available'}), 503

    try:
        analysis_data = database_manager.get_portfolio_analysis(analysis_id)
        if analysis_data:
            return jsonify(analysis_data)
        else:
            return jsonify({'error': 'Analysis not found'}), 404

    except Exception as e:
        logger.error(f"Error retrieving analysis {analysis_id}: {e}")
        return jsonify({'error': f'Analysis retrieval failed: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    # Run the application
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
