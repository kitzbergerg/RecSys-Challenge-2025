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

echo "Creating dataset..."
python -m embeddings_transformer.data_processor --data-dir /home/jovyan/shared/194.035-2025S/data/group_project/data_new/ --output-dir ../data/sequence/
echo "Make sure to switch between tasks (reconstruction/contrastive)!"
echo "Training transformer..."
python -m embeddings_transformer.model_training --data-dir /home/jovyan/shared/194.035-2025S/data/group_project/data_new/ --sequences-path ../data/sequence/ --task=reconstruction

echo "Make sure to set the CHECKPOINT_PATH env variable to the model after training!"
echo "Creating embeddings..."
python -m embeddings_transformer.create_embeddings --data-dir /home/jovyan/shared/194.035-2025S/data/group_project/data_new/ --embeddings-dir ../results/transformer/ --sequences-path ../data/sequence/ --checkpoint-path $CHECKPOINT_PATH

echo "Running validator..."
python -m validator.run --data-dir /home/jovyan/shared/194.035-2025S/data/group_project/data_new/ --embeddings-dir ../results/transformer/
