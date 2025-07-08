#!/bin/bash

echo "🚀 Starting Render deployment build..."

# Upgrade pip (no system packages needed on Render)
echo "⬆️ Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install Python dependencies with no cache to avoid issues
echo "📚 Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

# Create necessary directories
echo "🗂️ Creating necessary directories..."
mkdir -p data/cache
mkdir -p reports
mkdir -p backend/reports

# Set permissions
echo "🔐 Setting permissions..."
chmod -R 755 data/ || true
chmod -R 755 reports/ || true
chmod -R 755 backend/ || true

echo "✅ Build completed successfully!"
