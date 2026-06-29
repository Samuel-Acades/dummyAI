FROM python:3.11-slim

WORKDIR /code

# Install system dependencies needed for certain vector operations
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --default-timeout=100 --retries 5 -r /code/requirements.txt

# Pre-download the embedding model into the container image so it boots instantly
# (removed to speed up builds; model is cached via ./hf_cache volume at runtime)
COPY ./app /code/app

EXPOSE 8001

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]