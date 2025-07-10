

cd src

python3 ./autoencoder_pipeline.py --data-dir ../data/original/ --embeddings-dir ../embeddings

python -m validator.run --data-dir ../data/original/ --embeddings-dir ../embeddings
