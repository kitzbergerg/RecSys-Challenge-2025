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

echo "Creating dataset..."
python -m experiments.transformer.data_processor --data-dir ../data/original/ --output-dir ../data/sequence/
echo "Make sure to switch between tasks (reconstruction/contrastive)!"
echo "Training transformer..."
python -m experiments.transformer.model_training --data-dir ../data/original/ --sequences-path ../data/sequence/ --task=reconstruction

echo "Make sure to set the CHECKPOINT_PATH env variable to the model after training!"
echo "Creating embeddings..."
python -m experiments.transformer.create_embeddings --data-dir ../data/original/ --embeddings-dir ../results/transformer/ --sequences-path ../data/sequence/ --checkpoint-path $CHECKPOINT_PATH

echo "Running validator..."
python -m validator.run --data-dir ../data/original/ --embeddings-dir ../results/transformer/
