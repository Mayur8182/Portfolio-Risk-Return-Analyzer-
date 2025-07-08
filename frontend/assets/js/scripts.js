/**
 * Portfolio Risk Analysis Dashboard - Main JavaScript
 */

// Global variables
let currentPortfolioData = null;
let charts = {};

// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// Portfolio presets
const PORTFOLIO_PRESETS = {
    tech: {
        stocks: ['AAPL', 'GOOGL', 'MSFT'],
        weights: [40, 35, 25]
    },
    diversified: {
        stocks: ['SPY', 'QQQ', 'VTI', 'IWM'],
        weights: [40, 25, 20, 15]
    },
    growth: {
        stocks: ['TSLA', 'NVDA', 'AMD', 'NFLX'],
        weights: [30, 25, 25, 20]
    }
};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing app...');
    initializeApp();

    // Initialize dashboard if we're on the dashboard page
    if (window.location.pathname.includes('dashboard.html')) {
        initializeDashboard();
    }

    // Force setup event listeners after DOM is ready
    setTimeout(setupEventListeners, 100);
});

function initializeApp() {
    setupEventListeners();
    setupFormValidation();
    
    // Check if we have portfolio data in URL params or localStorage
    const urlParams = new URLSearchParams(window.location.search);
    const portfolioData = urlParams.get('portfolio');
    
    if (portfolioData) {
        try {
            const data = JSON.parse(decodeURIComponent(portfolioData));
            loadPortfolioData(data);
        } catch (error) {
            console.error('Error parsing portfolio data from URL:', error);
        }
    }
}

function setupEventListeners() {
    console.log('Setting up event listeners...');

    // Portfolio form submission
    const portfolioForm = document.getElementById('portfolioForm');
    if (portfolioForm) {
        console.log('Portfolio form found, adding submit listener');
        portfolioForm.addEventListener('submit', handlePortfolioSubmission);
    } else {
        console.log('Portfolio form NOT found!');
    }

    // Add stock button
    const addStockBtn = document.getElementById('addStock');
    if (addStockBtn) {
        console.log('Add stock button found, adding click listener');
        addStockBtn.addEventListener('click', addStockInput);
    } else {
        console.log('Add stock button NOT found!');
    }

    // Preset buttons
    const presetBtns = document.querySelectorAll('.preset-btn');
    console.log('Found preset buttons:', presetBtns.length);
    presetBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            console.log('Preset clicked:', this.dataset.preset);
            const preset = this.dataset.preset;
            loadPreset(preset);
        });
    });

    // Remove stock buttons (delegated event)
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('remove-stock') ||
            e.target.parentElement.classList.contains('remove-stock')) {
            console.log('Remove stock clicked');
            removeStockInput(e.target.closest('.input-group'));
        }
    });

    // Real-time weight validation
    document.addEventListener('input', function(e) {
        if (e.target.classList.contains('stock-weight')) {
            validateWeights();
        }
    });

    console.log('Event listeners setup complete');
}

function setupFormValidation() {
    // Add custom validation styles
    const forms = document.querySelectorAll('.needs-validation');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
}

function addStockInput() {
    console.log('Adding new stock input...');
    const stockInputs = document.getElementById('stockInputs');

    if (!stockInputs) {
        console.error('stockInputs element not found!');
        return;
    }

    const inputGroups = stockInputs.querySelectorAll('.input-group');

    if (inputGroups.length >= 10) {
        showAlert('Maximum 10 stocks allowed', 'warning');
        return;
    }

    const newInputGroup = document.createElement('div');
    newInputGroup.className = 'input-group mb-2';
    newInputGroup.innerHTML = `
        <input type="text" class="form-control stock-symbol" placeholder="e.g., AAPL" required>
        <input type="number" class="form-control stock-weight" placeholder="Weight %" min="0" max="100" step="0.1" required>
        <button type="button" class="btn btn-outline-danger remove-stock">
            <i class="fas fa-times"></i>
        </button>
    `;

    stockInputs.appendChild(newInputGroup);
    updateRemoveButtons();
    console.log('New stock input added successfully');
}

function removeStockInput(inputGroup) {
    inputGroup.remove();
    updateRemoveButtons();
    validateWeights();
}

function updateRemoveButtons() {
    const inputGroups = document.querySelectorAll('#stockInputs .input-group');
    inputGroups.forEach((group, index) => {
        const removeBtn = group.querySelector('.remove-stock');
        if (inputGroups.length > 1) {
            removeBtn.style.display = 'block';
        } else {
            removeBtn.style.display = 'none';
        }
    });
}

function loadPreset(presetName) {
    console.log('Loading preset:', presetName);
    const preset = PORTFOLIO_PRESETS[presetName];
    if (!preset) {
        console.error('Preset not found:', presetName);
        return;
    }

    // Clear existing inputs
    const stockInputs = document.getElementById('stockInputs');
    if (!stockInputs) {
        console.error('stockInputs element not found!');
        return;
    }

    stockInputs.innerHTML = '';

    // Add inputs for preset
    preset.stocks.forEach((stock, index) => {
        const inputGroup = document.createElement('div');
        inputGroup.className = 'input-group mb-2';
        inputGroup.innerHTML = `
            <input type="text" class="form-control stock-symbol" value="${stock}" required>
            <input type="number" class="form-control stock-weight" value="${preset.weights[index]}" min="0" max="100" step="0.1" required>
            <button type="button" class="btn btn-outline-danger remove-stock" style="display: ${preset.stocks.length > 1 ? 'block' : 'none'}">
                <i class="fas fa-times"></i>
            </button>
        `;
        stockInputs.appendChild(inputGroup);
    });

    validateWeights();
    console.log('Preset loaded successfully');
}

function validateWeights() {
    const weightInputs = document.querySelectorAll('.stock-weight');
    let totalWeight = 0;
    
    weightInputs.forEach(input => {
        const value = parseFloat(input.value) || 0;
        totalWeight += value;
    });
    
    // Update visual feedback
    const submitBtn = document.querySelector('#portfolioForm button[type="submit"]');
    if (submitBtn) {
        if (Math.abs(totalWeight - 100) > 1) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i>Weights must sum to 100% (currently ${totalWeight.toFixed(1)}%)`;
        } else {
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<i class="fas fa-chart-bar me-2"></i>Analyze Portfolio';
        }
    }
    
    return Math.abs(totalWeight - 100) <= 1;
}

async function handlePortfolioSubmission(event) {
    console.log('Portfolio form submitted!');
    event.preventDefault();

    if (!validateWeights()) {
        showAlert('Portfolio weights must sum to 100%', 'danger');
        return;
    }

    const formData = collectFormData();
    if (!formData) {
        showAlert('Please fill in all required fields', 'danger');
        return;
    }

    console.log('Form data collected:', formData);
    showLoading(true);

    // Update loading message for better user experience
    const loadingText = document.querySelector('#loadingSpinner .text-muted');
    if (loadingText) {
        loadingText.textContent = `Analyzing ${formData.stocks.length} stocks with institutional-grade accuracy...`;
    }

    try {
        console.log('Sending request to:', `${API_BASE_URL}/analyze`);

        // Create AbortController for timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 seconds timeout

        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        // Update loading message
        if (loadingText) {
            loadingText.textContent = 'Processing analysis results...';
        }

        console.log('Response status:', response.status);

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Analysis failed');
        }

        const data = await response.json();
        console.log('Analysis successful:', data);

        // Store data and redirect to dashboard
        localStorage.setItem('portfolioAnalysis', JSON.stringify(data));
        window.location.href = `dashboard.html?portfolio=${encodeURIComponent(JSON.stringify(formData))}`;

    } catch (error) {
        console.error('Error analyzing portfolio:', error);

        let errorMessage = 'Analysis failed';
        if (error.name === 'AbortError') {
            errorMessage = 'Analysis is taking longer than expected. Please wait and try again.';
        } else if (error.message.includes('Failed to fetch')) {
            errorMessage = 'Network connection error. Please check your connection and try again.';
        } else {
            errorMessage = `Analysis failed: ${error.message}`;
        }

        showAlert(errorMessage, 'danger');
    } finally {
        showLoading(false);
    }
}

function collectFormData() {
    const stockSymbols = Array.from(document.querySelectorAll('.stock-symbol'))
        .map(input => input.value.trim().toUpperCase())
        .filter(symbol => symbol);
    
    const stockWeights = Array.from(document.querySelectorAll('.stock-weight'))
        .map(input => parseFloat(input.value) || 0)
        .filter(weight => weight > 0);
    
    if (stockSymbols.length !== stockWeights.length || stockSymbols.length === 0) {
        return null;
    }
    
    // Convert percentages to decimals
    const normalizedWeights = stockWeights.map(weight => weight / 100);
    
    const analysisPeriod = document.getElementById('analysisPeriod')?.value || '1y';
    const riskFreeRate = parseFloat(document.getElementById('riskFreeRate')?.value || 2.0) / 100;
    
    return {
        stocks: stockSymbols,
        weights: normalizedWeights,
        period: analysisPeriod,
        risk_free_rate: riskFreeRate
    };
}

function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    const overlay = document.getElementById('loadingOverlay');
    
    if (spinner) {
        spinner.style.display = show ? 'block' : 'none';
    }
    
    if (overlay) {
        overlay.style.display = show ? 'flex' : 'none';
    }
}

function showAlert(message, type = 'info') {
    const alertContainer = document.getElementById('errorAlert');
    const messageElement = document.getElementById('errorMessage');
    
    if (alertContainer && messageElement) {
        messageElement.textContent = message;
        alertContainer.className = `alert alert-${type} mt-4`;
        alertContainer.style.display = 'block';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            alertContainer.style.display = 'none';
        }, 5000);
    } else {
        // Fallback to browser alert
        alert(message);
    }
}

// Utility functions
function formatCurrency(value, currency = 'USD') {
    if (typeof value !== 'number') return '--';
    
    if (Math.abs(value) >= 1e9) {
        return `$${(value / 1e9).toFixed(2)}B`;
    } else if (Math.abs(value) >= 1e6) {
        return `$${(value / 1e6).toFixed(2)}M`;
    } else if (Math.abs(value) >= 1e3) {
        return `$${(value / 1e3).toFixed(2)}K`;
    } else {
        return `$${value.toFixed(2)}`;
    }
}

function formatPercentage(value, decimals = 2) {
    if (typeof value !== 'number') return '--';
    return `${(value * 100).toFixed(decimals)}%`;
}

function formatNumber(value, decimals = 2) {
    if (typeof value !== 'number') return '--';
    return value.toFixed(decimals);
}

function getChangeClass(value) {
    if (value > 0) return 'positive';
    if (value < 0) return 'negative';
    return 'neutral';
}

function getChangeIcon(value) {
    if (value > 0) return 'fas fa-arrow-up';
    if (value < 0) return 'fas fa-arrow-down';
    return 'fas fa-minus';
}

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Initialize tooltips and popovers if Bootstrap is available
if (typeof bootstrap !== 'undefined') {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}
