

cd Rec-Sys-Challenge-Github-Repo

python3 ./autoencoder_pipeline.py --data-dir /home/jovyan/shared/194.035-2025S/data/group_project/data_new/ --embeddings-dir ../embeddings

python -m validator.run --data-dir /home/jovyan/shared/194.035-2025S/data/group_project/data_new/ --embeddings-dir ../embeddings
