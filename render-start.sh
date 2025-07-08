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
    --workers 2 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    app:app
