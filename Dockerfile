FROM python:3.11-slim

WORKDIR /code

# Install system dependencies needed for certain vector operations
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Pre-download the embedding model into the container image so it boots instantly
RUN python -c "from sentence_transformers import SentenceTransformer; Model = SentenceTransformer('all-MiniLM-L6-v2')"

COPY ./data /code/data
COPY ./app /code/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]