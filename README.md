# RecSys Challenge 2025 - Universal Behavioral Profiles

## Introduction

The goal was to create Universal Behavioral Profiles (i.e. embeddings) based on provided data (purchases, add to cart,
remove from cart,
page visits and search query).
Based on the embeddings a model is trained for multiple prediction tasks.
Well-designed embeddings should result in high scores for all tasks.

A detailed explanation of the challenge can be found [here](src/README.md) or on the
challenge [website](https://www.recsyschallenge.com/2025/).

## Dataset

To run experiments download the dataset (see [here](https://www.recsyschallenge.com/2025/), or use
the [direct link](https://data.recsys.synerise.com/dataset/ubc_data/ubc_data.tar.gz)).  
Extract the data to `data/original/` such that the directory `data/original/input` (and others) exists.

## Execution

To use our implementation, the following steps are needed. The best results were obtained using the autoencoder.

### Create embeddings using (baseline extended) calculators

To create the embeddings on the full set, start run_calculators.sh in shell.

```bash
./script/calculators.sh
```

### Create embeddings using the autoencoder

In `src/autoencoder_pipeline.py` you'll find a Config class where you can configure parameters. Included is a feature to
save and load features generated by the calculators, thereby saving time on embedding generation.

```bash
./scripts/autoencoder.sh
```

### Create embeddings using the contrastive learning

```bash
./scripts/contrastive.sh
```

### Create embeddings using transformer

Training the transformer takes a long time (>15h on Radeon RX 7900 XTX). It also might require some manual tuning of
parameters like masking probabilities in training dataset.

`transformer.sh` gives a general idea of what has to be done. Note that just running this file won't work, as the
embeddings generation step requires setting the path to pytorch_lighting checkpoints.

```bash
./scripts/transformer.sh
```
