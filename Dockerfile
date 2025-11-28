# Use small official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Expose port (optional)
EXPOSE 5000

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Start app: run migrations before Gunicorn
CMD ["bash", "-lc", "flask db upgrade && gunicorn -w 4 -b 0.0.0.0:${PORT:-5000} app:app -k gthread"]

