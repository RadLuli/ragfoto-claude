#!/bin/bash
# Script for deploying the photo assessment application

# Set error handling
set -e

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install it first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Please install it first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Please install it from https://ollama.ai/"
    echo "Then run: ollama pull llama3"
    exit 1
fi

# Create data directories if they don't exist
echo "Creating data directories..."
mkdir -p data/pdfs
mkdir -p data/ebooks
mkdir -p data/web_content
mkdir -p data/vectordb

# Check if there are PDFs in the data directory
pdf_count=$(find data/pdfs -name "*.pdf" | wc -l)
epub_count=$(find data/ebooks -name "*.epub" | wc -l)

if [ "$pdf_count" -eq 0 ] && [ "$epub_count" -eq 0 ]; then
    echo "Warning: No PDF or EPUB files found in data directories."
    echo "Please add your reference materials before running the application."
fi

# Process documents to create vector database
echo "Processing documents to create vector database..."
python -c "from src.document_processing.process_documents import process_documents; process_documents('config.json')"

# Launch app with Streamlit
echo "Starting the application..."
streamlit run app.py
