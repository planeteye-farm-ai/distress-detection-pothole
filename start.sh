#!/bin/bash
# 🚀 Start Flask + SocketIO server using Gunicorn + Eventlet
set -e

# Activate virtual environment
source .venv/bin/activate

# Log environment info (optional)
echo "=== Starting Distress Detection Flask server ==="
echo "Python version: $(python --version)"
echo "PORT: ${PORT:-5000}"

# Start Gunicorn with Eventlet worker
# Single worker prevents memory overload on Render (2GB RAM plan)
gunicorn --worker-class eventlet --workers 1 --threads 2 \
  --timeout 600 --bind 0.0.0.0:${PORT:-5000} app:app
