python3 -m pip install -r requirements.txt

cd Rec-Sys-Challenge-Github-Repo

python -m custom_pipline.create_embeddings --data-dir /home/jovyan/shared/194.035-2025S/data/group_project/data_new/ --embeddings-dir ../embeddings_split --split

python -m training_pipeline.train --data-dir /home/jovyan/shared/194.035-2025S/data/group_project/data --embeddings-dir ../embeddings_split --tasks churn propensity_category propensity_sku --log-name exp_20250401 --accelerator gpu --devices auto --disable-relevant-clients-check --score-dir ../scores --neptune-project /home/jovyan/groups/194.035-2025S/Group_36/Group_36/neptune