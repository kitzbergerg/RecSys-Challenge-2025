## Run

```shell
python -m experiments.transformer.data_processor --data-dir ../data/original/ --output-dir ../data/sequence/
python -m experiments.transformer.model_training --data-dir ../data/original/ --output-dir ../models/ --sequences-path ../data/sequence/

python -m experiments.transformer.create_embeddings --data-dir ../data/original/ --embeddings-dir ../results/transformer/v2/ --sequences-path ../data/sequence/ --checkpoint-path lightning_logs/version_3/checkpoints/epoch=29-step=210960.ckpt
python -m validator.run --data-dir ../data/original/ --embeddings-dir ../results/transformer/v1/

python -m training_pipeline.train --data-dir ../data/original/ --embeddings-dir ../results/transformer/v2/ --tasks churn propensity_category propensity_sku --log-name baseline --accelerator gpu --devices 0 --disable-relevant-clients-check
```
