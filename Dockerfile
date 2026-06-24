FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install yt-dlp
RUN curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp \
    && chmod a+rx /usr/local/bin/yt-dlp

WORKDIR /app

# Copy and install Python deps first (layer caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .[all]

# Copy source
COPY src/ ./src/
COPY migrations/ ./migrations/

# Create runtime dirs
RUN mkdir -p artifacts media_cache

EXPOSE 8000

CMD ["uvicorn", "fact_checker.api:app", "--host", "0.0.0.0", "--port", "8000"]
