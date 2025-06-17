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

echo "Running calculators pipeline with split data..."
python -m custom_pipline.create_embeddings --data-dir /home/jovyan/shared/194.035-2025S/data/group_project/data_new/ --embeddings-dir ../embeddings_split --split

echo "Running training pipeline with split data..."
python -m training_pipeline.train --data-dir /home/jovyan/shared/194.035-2025S/data/group_project/data_new/ --embeddings-dir ../embeddings_split --tasks churn propensity_category propensity_sku --log-name exp_20250401 --accelerator gpu --devices auto --disable-relevant-clients-check --score-dir ../scores --neptune-project /home/jovyan/groups/194.035-2025S/Group_36/Group_36/neptune