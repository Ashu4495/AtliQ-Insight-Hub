# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8080

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the spacy models needed for the application
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_md

# Pre-download the embedding model to avoid runtime downloads
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p $HF_HOME
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')"

# Copy the rest of the application code
COPY . .

# IMPORTANT: Build the Knowledge Base (ChromaDB) during the Docker build
# This ensures a clean, compatible database is baked INTO the image.
# We set PYTHONPATH to include the current directory so src is importable.
RUN PYTHONPATH=. python src/ingestion/vectorstore/chroma_store.py

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["python", "src/app.py"]
