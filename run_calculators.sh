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

echo "Changing to the Rec-Sys-Challenge-Github-Repo directory..."
cd Rec-Sys-Challenge-Github-Repo

echo "Running contrastive pipeline..."
python -m custom_pipline.create_embeddings --data-dir /home/jovyan/shared/194.035-2025S/data/group_project/data_new/ --embeddings-dir ../embeddings

echo "Running validator..."
python -m validator.run --data-dir /home/jovyan/shared/194.035-2025S/data/group_project/data_new/ --embeddings-dir ../embeddings