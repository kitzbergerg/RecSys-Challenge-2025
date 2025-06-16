# Universal Behavioral Modeling Data Challenge

## Introduction
The goal was to create Universal Behavioral Profile, which is based on provided data (purchases, add to cart, remove from cart, page visits and search query). This data is extracted and used to create a set of features for each client this is a user representation which is then used for training a model for multiple prediction tasks.

## Execution
To use our implementation, the following steps are needed. 

### Run locally
To run the implementation locally, just start the run_locally.sh file. The embeddings will be created based on the split dataset and the training on the new embeddings is executed.

### Create embeddings on full dataset
To create the embeddings on the full set, start run_standard_embeddings.sh in shell. The embeddings will be stored in an embeddings folder.

### Create embeddings using the autoencoder:

Run run_autoencoder.sh, the python file also accepts --data-dir and --embeddings-dir CLI arguments for custom in and out locations.
If you open the Rec-Sys-Challenge-Github-Repo/autoencoder_pipeline.py file, you'll find a Config class at the top where you can configure a bunch of parameters as well.






## 
Add your Documentation here. The description can be found on [TUWEL](https://tuwel.tuwien.ac.at/mod/page/view.php?id=2495021)

The folder Rec-Sys-Challenge-Github-Repo contains the [Synerise RecSys Challenge 2025 Github-Repo](https://github.com/Synerise/recsys2025) (accessed 20. 4. 2025)