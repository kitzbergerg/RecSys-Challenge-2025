#!/bin/bash

# Exit script if any command fails
set -e

echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies from requirements..."
pip install -r src/requirements.in

echo "Changing to the src directory..."
cd src

echo "Running contrastive pipeline..."
python -m experiments.contrastive_embeddings.contrastive_enhanced --data-dir ../data/original/ --embeddings-dir ../results

echo "Running validator..."
python -m validator.run --data-dir ../data/original/ --embeddings-dir ../results