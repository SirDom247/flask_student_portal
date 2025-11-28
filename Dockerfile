# Use small official Python image
FROM python:3.11-slim

# set workdir
WORKDIR /app

# system deps, if any (adjust)
RUN apt-get update && apt-get install -y build-essential libpq-dev --no-install-recommends && rm -rf /var/lib/apt/lists/*

# copy requirements first to leverage caching
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# copy app
COPY . .

# expose not required but nice
EXPOSE 10000

# create a non-root user (optional)
RUN useradd -m appuser
USER appuser

# Use bash -lc so $PORT expands at runtime
CMD ["bash", "-lc", "gunicorn -w 4 -b 0.0.0.0:${PORT:-10000} 'app:create_app()' -k gthread"]
