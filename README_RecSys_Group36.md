# Universal Behavioral Modeling Data Challenge

## Introduction
The goal was to create Universal Behavioral Profile, which is based on provided data (purchases, add to cart, remove from cart, page visits and search query). This data is extracted and used to create a set of features for each client this is a user representation which is then used for training a model for multiple prediction tasks.

## Leaderboard embeddings
The embeddings uploaded in the leaderboard can be found in the submission folder (gitlab). They were obtained using the autoencoder model (the same which can be run under run.sh)
For the other models, the last available embeddings are provided in the respective folders within embeddings_provided (jupyerhub)

## Execution
To use our implementation, the following steps are needed. The best results so far were obtained using the autoencoder.

### Create embeddings on full dataset using (baseline extended) calculators
To create the embeddings on the full set, start run_calculators.sh in shell. The embeddings will be stored in the embeddings folder.
```bash
run_calculators.sh
```

#### Run locally using (baseline extended) calculators
To run the implementation locally with the calculators from the extended baseline (custom pipeline), just start the run_locally.sh file. The embeddings will be created based on the split dataset and the training on the new embeddings is executed.

```bash
run_locally.sh
```

### Create embeddings using the autoencoder
Run run.sh, the python file also accepts --data-dir and --embeddings-dir CLI arguments for custom in and out locations, which are predefined in the run.sh file.
If you open the Rec-Sys-Challenge-Github-Repo/autoencoder_pipeline.py file, you'll find a Config class at the top where you can configure a bunch of parameters as well. Included is a feature to save the raw features after generation and to then load them on subsequent runs to skip the lenghthy generation process, as that's something that only needs to be done if something in the underlying feature engineering phase has changed.



```bash
run.sh
```

After running, we evaluate the models on the training pipeline provided by the organizers, obtaining AUROC scores for the three non-hidden categories.

This can be done by running 
```bash
python -m training_pipeline.train --data-dir /home/jovyan/shared/194.035-2025S/data/group_project/data_new --embeddings-dir /home/jovyan/groups/194.035-2025S/Group_36/Group_36/Rec-Sys-Challenge-Github-Repo/output_new --tasks churn propensity_category propensity_sku --log-name exp_20250401 --accelerator gpu --devices auto --disable-relevant-clients-check --score-dir /home/jovyan/groups/194.035-2025S/Group_36/Group_36/scores --neptune-project /home/jovyan/groups/194.035-2025S/Group_36/Group_36/neptune
```
from the Rec-Sys-Challenge-Github-Repo folder.

You'll then find the scores in the scores folder!


```bash
run.sh
```

### Create embeddings using the contrastive learning

The `run_contrastive.sh` script sets up the prerequisites, generates client embeddings using the `contrastive_enhanced.py` script, and then validates the generated embeddings.

To execute the script, simply run it from your terminal:

```bash
./run_contrastive.sh
```

### Create embeddings using transformer

```bash
run_transformer.sh
```

## Metrics local training
In the `metrics.ipynb` the auroc, diversity, novelty are visible for the local training of the embeddings with split.    