#!/bin/bash

# Exit script if any command fails
set -e

echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Changing to the src directory..."
cd src

echo "Running autoencoder pipeline..."
python3 ./autoencoder_pipeline.py \
  --data-dir ../data/original/ \
  --embeddings-dir ../embeddings

echo "Running validator..."
python -m validator.run \
  --data-dir ../data/original/ \
  --embeddings-dir ../embeddings

echo "All tasks completed successfully."