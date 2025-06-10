FROM python:3.11-slim

# Install system dependencies for pdfplumber (poppler-utils) and pypdfium2
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK punkt and all data at build time
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader all

# Copy script and sample PDFs
COPY extract_and_build_dictionary.py ./
COPY sample ./sample

# Set default command
CMD ["python", "extract_and_build_dictionary.py"] 