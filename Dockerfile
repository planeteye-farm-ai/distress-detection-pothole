# Use lightweight Python base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install required system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies (no cache to save space)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Ensure uploads and model directories exist
RUN mkdir -p /app/uploads /app/data/models

# Attempt to download SAM model if missing (optional)
RUN echo "⬇️ Checking SAM model in /app/data/models..." && \
    if [ ! -f "/app/data/models/sam_vit_b_01ec64.pth" ]; then \
        echo "Downloading SAM model..."; \
        curl -L -o /app/data/models/sam_vit_b_01ec64.pth \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
        || echo "⚠️ SAM model download failed — will attempt at runtime."; \
    else \
        echo "✅ SAM model already present in /app/data/models/"; \
    fi

# Environment variables
ENV PORT=8080
ENV FLASK_ENV=production
ENV MODEL_DIR=/app/data/models
ENV UPLOAD_FOLDER=/app/uploads
ENV PYTHONUNBUFFERED=1

# Expose default port (Cloud Run injects $PORT)
EXPOSE 8080

# Start Flask-SocketIO via Gunicorn + Eventlet
CMD ["gunicorn", "-k", "eventlet", "-w", "1", "--threads", "1", "--timeout", "600", "--bind", "0.0.0.0:$PORT", "app:app"]
