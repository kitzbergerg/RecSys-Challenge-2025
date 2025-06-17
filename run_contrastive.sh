python3 -m pip install -r requirements.txt

cd Rec-Sys-Challenge-Github-Repo

python3 contrastive_embeddings/contrastive_enhanced.py --data-dir /home/jovyan/shared/194.035-2025S/data/group_project/data_new/ --embeddings-dir ../embeddings

python -m validator.run --data-dir /home/jovyan/shared/194.035-2025S/data/group_project/data_new/ --embeddings-dir ../embedd