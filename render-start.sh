#!/bin/bash

# Render Start Script for Portfolio Analytics Platform
echo "🚀 Starting Portfolio Analytics Platform on Render..."

# Set environment variables for production
export FLASK_ENV=production
export FLASK_DEBUG=false
export PYTHONPATH="${PYTHONPATH}:$(pwd)/backend"

# Navigate to backend directory
cd backend

# Start the application with Gunicorn
echo "🌐 Starting Gunicorn server..."
exec gunicorn \
    --bind 0.0.0.0:$PORT \
    --workers 4 \
    --worker-class gevent \
    --worker-connections 1000 \
    --timeout 120 \
    --keep-alive 5 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --preload \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    app:app
