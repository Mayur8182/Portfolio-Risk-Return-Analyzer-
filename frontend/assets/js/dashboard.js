/**
 * Portfolio Dashboard - Chart rendering and data visualization
 */

// Dashboard-specific variables
let portfolioAnalysisData = null;
let dashboardCharts = {};

// Chart.js default configuration
Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.color = '#6b7280';

function initializeDashboard() {
    loadPortfolioAnalysisData();
    setupDashboardEventListeners();
}

function loadPortfolioAnalysisData() {
    // Try to load from localStorage first
    const storedData = localStorage.getItem('portfolioAnalysis');
    if (storedData) {
        try {
            portfolioAnalysisData = JSON.parse(storedData);
            renderDashboard();
            return;
        } catch (error) {
            console.error('Error parsing stored portfolio data:', error);
        }
    }
    
    // If no stored data, show message to analyze portfolio first
    showDashboardMessage('No portfolio data found. Please analyze a portfolio first.', 'warning');
}

function renderDashboard() {
    if (!portfolioAnalysisData || !portfolioAnalysisData.success) {
        showDashboardMessage('Invalid portfolio data. Please analyze a portfolio again.', 'danger');
        return;
    }
    
    const data = portfolioAnalysisData.data;
    
    // Render key metrics
    renderKeyMetrics(data.risk_metrics, data.performance_metrics);
    
    // Render portfolio summary
    renderPortfolioSummary(data.portfolio);
    
    // Render sentiment overview
    renderSentimentOverview(data.sentiment);
    
    // Render charts
    renderAllCharts(data.charts);
    
    // Render risk metrics table
    renderRiskMetricsTable(data.risk_metrics);
}

function renderKeyMetrics(riskMetrics, performanceMetrics) {
    // Total Return
    const totalReturnElement = document.getElementById('totalReturn');
    if (totalReturnElement && performanceMetrics.portfolio_total_return !== undefined) {
        totalReturnElement.textContent = formatPercentage(performanceMetrics.portfolio_total_return);
        totalReturnElement.className = `metric-value ${getChangeClass(performanceMetrics.portfolio_total_return)}`;
    }
    
    // Sharpe Ratio
    const sharpeElement = document.getElementById('sharpeRatio');
    if (sharpeElement && riskMetrics.sharpe_ratio !== undefined) {
        sharpeElement.textContent = formatNumber(riskMetrics.sharpe_ratio, 3);
        sharpeElement.className = `metric-value ${getChangeClass(riskMetrics.sharpe_ratio)}`;
    }
    
    // Volatility
    const volatilityElement = document.getElementById('volatility');
    if (volatilityElement && riskMetrics.annualized_volatility !== undefined) {
        volatilityElement.textContent = formatPercentage(riskMetrics.annualized_volatility);
        volatilityElement.className = `metric-value ${getChangeClass(-riskMetrics.annualized_volatility)}`;
    }
    
    // Max Drawdown
    const drawdownElement = document.getElementById('maxDrawdown');
    if (drawdownElement && riskMetrics.max_drawdown !== undefined) {
        drawdownElement.textContent = formatPercentage(riskMetrics.max_drawdown);
        drawdownElement.className = `metric-value ${getChangeClass(riskMetrics.max_drawdown)}`;
    }
}

function renderPortfolioSummary(portfolio) {
    const summaryElement = document.getElementById('portfolioSummary');
    if (!summaryElement || !portfolio) return;
    
    const summaryHTML = `
        <div class="mb-3">
            <h6 class="text-light mb-2">Assets (${portfolio.stocks.length})</h6>
            ${portfolio.stocks.map((stock, index) => `
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <span class="text-light">${stock}</span>
                    <span class="badge bg-light text-dark">${formatPercentage(portfolio.weights[index])}</span>
                </div>
            `).join('')}
        </div>
        <div class="text-center">
            <small class="text-light opacity-75">Total Weight: ${formatPercentage(portfolio.total_value)}</small>
        </div>
    `;
    
    summaryElement.innerHTML = summaryHTML;
}

function renderSentimentOverview(sentiment) {
    const sentimentElement = document.getElementById('sentimentOverview');
    if (!sentimentElement || !sentiment) return;
    
    const portfolioSentiment = sentiment.portfolio_sentiment;
    const sentimentClass = getSentimentClass(portfolioSentiment.overall_score);
    
    const sentimentHTML = `
        <div class="text-center mb-3">
            <div class="h4 text-light mb-1">${portfolioSentiment.sentiment_label}</div>
            <div class="small text-light opacity-75">Score: ${formatNumber(portfolioSentiment.overall_score, 2)}</div>
        </div>
        <div class="progress mb-2" style="height: 8px;">
            <div class="progress-bar ${sentimentClass}" 
                 style="width: ${Math.abs(portfolioSentiment.overall_score) * 100}%"></div>
        </div>
        <small class="text-light opacity-75">
            Confidence: ${formatPercentage(portfolioSentiment.confidence)}
        </small>
    `;
    
    sentimentElement.innerHTML = sentimentHTML;
}

function renderAllCharts(chartsData) {
    if (!chartsData) return;
    
    // Portfolio Performance Chart
    if (chartsData.portfolio_performance) {
        renderChart('performanceChart', chartsData.portfolio_performance);
    }
    
    // Asset Allocation Chart
    if (chartsData.asset_allocation) {
        renderChart('allocationChart', chartsData.asset_allocation);
    }
    
    // Risk Radar Chart
    if (chartsData.risk_radar) {
        renderChart('riskRadarChart', chartsData.risk_radar);
    }
    
    // Individual Performance Chart
    if (chartsData.individual_performance) {
        renderChart('individualPerformanceChart', chartsData.individual_performance);
    }
    
    // Returns Distribution Chart
    if (chartsData.returns_distribution) {
        renderChart('returnsDistributionChart', chartsData.returns_distribution);
    }
    
    // Correlation Heatmap
    if (chartsData.correlation_heatmap) {
        renderCorrelationHeatmap(chartsData.correlation_heatmap);
    }
}

function renderChart(canvasId, chartConfig) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    // Destroy existing chart if it exists
    if (dashboardCharts[canvasId]) {
        dashboardCharts[canvasId].destroy();
    }
    
    const ctx = canvas.getContext('2d');
    
    // Apply responsive configuration
    const config = {
        ...chartConfig,
        options: {
            ...chartConfig.options,
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                ...chartConfig.options?.plugins,
                legend: {
                    ...chartConfig.options?.plugins?.legend,
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                }
            }
        }
    };
    
    dashboardCharts[canvasId] = new Chart(ctx, config);
}

function renderCorrelationHeatmap(heatmapData) {
    const container = document.getElementById('correlationHeatmap');
    if (!container || !heatmapData.data) return;
    
    const labels = heatmapData.labels;
    const size = labels.length;
    
    // Create heatmap HTML
    let heatmapHTML = '<div class="correlation-heatmap" style="grid-template-columns: repeat(' + size + ', 1fr);">';
    
    heatmapData.data.forEach(cell => {
        const intensity = Math.abs(cell.v);
        const hue = cell.v >= 0 ? 120 : 0; // Green for positive, red for negative
        const backgroundColor = `hsla(${hue}, 70%, 50%, ${intensity})`;
        
        heatmapHTML += `
            <div class="correlation-cell" 
                 style="background-color: ${backgroundColor};"
                 title="${cell.row_label} vs ${cell.col_label}: ${cell.v.toFixed(3)}">
                ${cell.v.toFixed(2)}
            </div>
        `;
    });
    
    heatmapHTML += '</div>';
    
    // Add labels
    heatmapHTML += '<div class="mt-2 d-flex justify-content-between small text-muted">';
    labels.forEach(label => {
        heatmapHTML += `<span>${label}</span>`;
    });
    heatmapHTML += '</div>';
    
    container.innerHTML = heatmapHTML;
}

function renderRiskMetricsTable(riskMetrics) {
    const tableBody = document.querySelector('#riskMetricsTable tbody');
    if (!tableBody || !riskMetrics) return;
    
    const metrics = [
        {
            name: 'Sharpe Ratio',
            value: formatNumber(riskMetrics.sharpe_ratio, 3),
            interpretation: getSharpeInterpretation(riskMetrics.sharpe_ratio),
            benchmark: '> 1.0'
        },
        {
            name: 'Sortino Ratio',
            value: formatNumber(riskMetrics.sortino_ratio, 3),
            interpretation: getSortinoInterpretation(riskMetrics.sortino_ratio),
            benchmark: '> 1.0'
        },
        {
            name: 'Volatility (Annual)',
            value: formatPercentage(riskMetrics.annualized_volatility),
            interpretation: getVolatilityInterpretation(riskMetrics.annualized_volatility),
            benchmark: '< 20%'
        },
        {
            name: 'Value at Risk (95%)',
            value: formatPercentage(riskMetrics.var_95),
            interpretation: getVaRInterpretation(riskMetrics.var_95),
            benchmark: '> -5%'
        },
        {
            name: 'Maximum Drawdown',
            value: formatPercentage(riskMetrics.max_drawdown),
            interpretation: getDrawdownInterpretation(riskMetrics.max_drawdown),
            benchmark: '> -20%'
        },
        {
            name: 'Beta',
            value: formatNumber(riskMetrics.beta, 2),
            interpretation: getBetaInterpretation(riskMetrics.beta),
            benchmark: '0.8 - 1.2'
        }
    ];
    
    tableBody.innerHTML = metrics.map(metric => `
        <tr>
            <td class="fw-semibold">${metric.name}</td>
            <td>${metric.value}</td>
            <td>${metric.interpretation}</td>
            <td class="text-muted">${metric.benchmark}</td>
        </tr>
    `).join('');
}

function setupDashboardEventListeners() {
    // Refresh data button
    const refreshBtn = document.querySelector('[onclick="refreshData()"]');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', refreshData);
    }
    
    // Optimize portfolio button
    const optimizeBtn = document.querySelector('[onclick="optimizePortfolio()"]');
    if (optimizeBtn) {
        optimizeBtn.addEventListener('click', optimizePortfolio);
    }
    
    // Export report button
    const exportBtn = document.querySelector('[onclick="exportReport()"]');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportReport);
    }
}

// Dashboard action functions
function refreshData() {
    showLoading(true);
    
    // Simulate data refresh
    setTimeout(() => {
        renderDashboard();
        showLoading(false);
        showAlert('Data refreshed successfully', 'success');
    }, 2000);
}

async function optimizePortfolio() {
    if (!portfolioAnalysisData) return;
    
    showLoading(true);
    
    try {
        const portfolio = portfolioAnalysisData.data.portfolio;
        const response = await fetch(`${API_BASE_URL}/optimize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                stocks: portfolio.stocks,
                objective: 'max_sharpe'
            })
        });
        
        if (!response.ok) {
            throw new Error('Optimization failed');
        }
        
        const optimizationData = await response.json();
        showOptimizationResults(optimizationData);
        
    } catch (error) {
        console.error('Error optimizing portfolio:', error);
        showAlert('Portfolio optimization failed', 'danger');
    } finally {
        showLoading(false);
    }
}

function showOptimizationResults(data) {
    const modal = new bootstrap.Modal(document.getElementById('optimizationModal'));
    const resultsContainer = document.getElementById('optimizationResults');
    
    if (data.max_sharpe && data.max_sharpe.success) {
        const result = data.max_sharpe;
        resultsContainer.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <h6>Optimized Weights</h6>
                    <div class="list-group">
                        ${result.weights.map((weight, index) => `
                            <div class="list-group-item d-flex justify-content-between">
                                <span>${portfolioAnalysisData.data.portfolio.stocks[index]}</span>
                                <span class="fw-bold">${formatPercentage(weight)}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
                <div class="col-md-6">
                    <h6>Expected Metrics</h6>
                    <table class="table table-sm">
                        <tr>
                            <td>Expected Return</td>
                            <td class="fw-bold">${formatPercentage(result.metrics.expected_return)}</td>
                        </tr>
                        <tr>
                            <td>Volatility</td>
                            <td class="fw-bold">${formatPercentage(result.metrics.volatility)}</td>
                        </tr>
                        <tr>
                            <td>Sharpe Ratio</td>
                            <td class="fw-bold">${formatNumber(result.metrics.sharpe_ratio, 3)}</td>
                        </tr>
                    </table>
                </div>
            </div>
        `;
    } else {
        resultsContainer.innerHTML = '<div class="alert alert-warning">Optimization failed or returned no results.</div>';
    }
    
    modal.show();
}

function exportReport() {
    if (!portfolioAnalysisData) {
        showAlert('No portfolio data available for export', 'danger');
        return;
    }

    // Get the button that was clicked
    const button = document.querySelector('[onclick="exportReport()"]');
    if (!button) return;

    // Show loading indicator
    const originalText = button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Generating PDF...';
    button.disabled = true;

    // Send data to PDF export endpoint
    fetch('http://localhost:5000/api/export-pdf', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            analysis_data: portfolioAnalysisData.data
        })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.error || 'PDF generation failed');
            });
        }
        return response.blob();
    })
    .then(blob => {
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `portfolio_analysis_${new Date().toISOString().split('T')[0]}.pdf`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);

        showAlert('PDF report exported successfully!', 'success');
    })
    .catch(error => {
        console.error('PDF export error:', error);
        showAlert(`PDF export failed: ${error.message}`, 'danger');
    })
    .finally(() => {
        // Restore button
        button.innerHTML = originalText;
        button.disabled = false;
    });
}

function switchToVisualDashboard() {
    if (!portfolioAnalysisData) {
        showAlert('No portfolio data available. Please analyze a portfolio first.', 'danger');
        return;
    }

    // Get current portfolio data from URL or stored data
    const urlParams = new URLSearchParams(window.location.search);
    const portfolioParam = urlParams.get('portfolio');

    if (portfolioParam) {
        // Redirect to visual dashboard with same portfolio data
        window.location.href = `visual-dashboard.html?portfolio=${portfolioParam}`;
    } else {
        // Fallback: redirect to visual dashboard and let it load from localStorage
        window.location.href = 'visual-dashboard.html';
    }
}

// Utility functions for interpretations
function getSharpeInterpretation(value) {
    if (value > 2) return 'Excellent';
    if (value > 1) return 'Good';
    if (value > 0) return 'Acceptable';
    return 'Poor';
}

function getSortinoInterpretation(value) {
    if (value > 2) return 'Excellent';
    if (value > 1) return 'Good';
    if (value > 0) return 'Acceptable';
    return 'Poor';
}

function getVolatilityInterpretation(value) {
    if (value < 0.1) return 'Low Risk';
    if (value < 0.2) return 'Moderate Risk';
    if (value < 0.3) return 'High Risk';
    return 'Very High Risk';
}

function getVaRInterpretation(value) {
    if (value > -0.02) return 'Low Risk';
    if (value > -0.05) return 'Moderate Risk';
    if (value > -0.1) return 'High Risk';
    return 'Very High Risk';
}

function getDrawdownInterpretation(value) {
    if (value > -0.05) return 'Excellent';
    if (value > -0.1) return 'Good';
    if (value > -0.2) return 'Acceptable';
    return 'Poor';
}

function getBetaInterpretation(value) {
    if (value < 0.8) return 'Defensive';
    if (value <= 1.2) return 'Market-like';
    return 'Aggressive';
}

function getSentimentClass(score) {
    if (score > 0.1) return 'bg-success';
    if (score < -0.1) return 'bg-danger';
    return 'bg-warning';
}

function showDashboardMessage(message, type) {
    const mainContent = document.querySelector('.col-lg-9');
    if (mainContent) {
        mainContent.innerHTML = `
            <div class="alert alert-${type} text-center">
                <h4>${message}</h4>
                <a href="index.html" class="btn btn-primary mt-3">Analyze Portfolio</a>
            </div>
        `;
    }
}
