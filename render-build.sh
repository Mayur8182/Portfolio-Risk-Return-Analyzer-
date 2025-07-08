#!/bin/bash

# Render Build Script for Portfolio Analytics Platform
echo "🚀 Starting Render deployment build..."

# Update system packages
echo "📦 Updating system packages..."
apt-get update

# Install system dependencies for financial libraries
echo "🔧 Installing system dependencies..."
apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    tcl8.6-dev \
    tk8.6-dev \
    python3-tk \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1-dev

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Install additional production dependencies
echo "🏭 Installing production dependencies..."
pip install gunicorn==21.2.0
pip install gevent==23.7.0
pip install psutil==5.9.5

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p reports
mkdir -p logs
mkdir -p static/uploads

# Set permissions
echo "🔐 Setting permissions..."
chmod +x backend/app.py

# Verify installation
echo "✅ Verifying installation..."
python -c "import flask, pandas, numpy, yfinance, pymongo; print('All core dependencies installed successfully!')"

echo "🎉 Build completed successfully!"
