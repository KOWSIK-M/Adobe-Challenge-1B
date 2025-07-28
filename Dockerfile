# Use a compatible base image
FROM --platform=linux/amd64 python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install \
        "numpy<2" \
        "huggingface_hub==0.13.4" \
        "torch==2.2.2+cpu" \
        "torchvision==0.17.2+cpu" \
        "torchaudio==2.2.2+cpu" \
        --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install "sentence-transformers==2.2.2" "PyMuPDF==1.22.5" && \
    rm -rf ~/.cache/pip

# Create folders if not mounted
RUN mkdir -p input output

# Default command
CMD ["python", "main.py"]
