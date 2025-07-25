import argparse
import logging
from typing import List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from IPython.display import display

from experiments.autoencoder.constants import (
    EVENT_TYPE_TO_COLUMNS, EventTypes,
)
from data_utils.utils import (
    load_with_properties,
    join_properties
)
from data_utils.data_dir import DataDir
from experiments.autoencoder.features_aggregator import (
    FeaturesAggregator,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def load_relevant_clients_ids(input_dir: Path) -> np.ndarray:
    return np.load(input_dir / "relevant_clients.npy")


def save_embeddings(
    embeddings_dir: Path, embeddings: np.ndarray, client_ids: np.ndarray
):
    """
    Function creates embeddings directory and saves embeddings in competition entry format.

    Args:
    embeddings_dir (Path): The directory where to save embeddings and client_ids.
    embeddings (np.ndarray): 2-d array storing embeddings.
    client_ids (np.ndarray): 1-d array storing client_ids corresponding to vectors from embeddings array.
    """
    logger.info("Saving embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_dir / "embeddings.npy", embeddings)
    np.save(embeddings_dir / "client_ids.npy", client_ids)


def create_embeddings(
    data_dir: DataDir,
    num_days: List[int],
    top_n: int,
    relevant_client_ids: np.ndarray,
    split: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate and merge user representation embeddings for specified event types.

    This function processes event data from CSV files, aggregates user's events based
    on specified columns for each event type, and merges these embeddings into a single
    user representation.

    Args:
        data_dir (DataDir): The DataDir class where Paths to raw event data, input and targte folders are stored.
        num_days (List[int]): A list of time windows (in days) for generating features.
        Each time window will produce different set of features from aggregated events
        from defined period.
        top_n (int): Number of columns' top values to consider for aggregating events.

    Returns:
        Tuple[np.ndarray, np.ndarray] : generated feature matrix and the list of all
        clients in two np.ndarray's.
    """
    aggregator = FeaturesAggregator(
        num_days=num_days,
        top_n=top_n,
        relevant_client_ids=relevant_client_ids,
    )

    logger.info("Loading product properties...")
    product_properties = pd.read_parquet(data_dir.data_dir / "product_properties.parquet")

    logger.info("Loading buy events...")
    #buy_df = pd.read_parquet(data_dir.data_dir / "product_buy.parquet")
    #properties_df = pd.read_parquet(data_dir.properties_file)
    #buy_df = join_properties(event_df=buy_df, properties_df=properties_df)
    buy_df = load_with_properties(data_dir=data_dir, event_type=EventTypes.PRODUCT_BUY.value, split=split)
    #buy_df = load_with_properties(data_dir, event_type=EventTypes.PRODUCT_BUY.value)
    buy_df["timestamp"] = pd.to_datetime(buy_df["timestamp"])
    display(buy_df)
    for event_type in EVENT_TYPE_TO_COLUMNS.keys():
        logger.info("Generating features for %s event type", event_type.value)
        logger.info("Loading data...")
        event_df = load_with_properties(data_dir=data_dir, event_type=event_type.value, split=split)
        event_df["timestamp"] = pd.to_datetime(event_df.timestamp)
        display(event_df)
        logger.info("Generating features...")
        aggregator.generate_features(
            event_type=event_type,
            client_id_column="client_id",
            df=event_df,
            columns=EVENT_TYPE_TO_COLUMNS[event_type],
            product_properties=product_properties,
            buy_events=buy_df if event_type == EventTypes.ADD_TO_CART else None,
        )

    logger.info("Merging features into embeddings")
    client_ids, embeddings = aggregator.merge_features()
    return client_ids, embeddings


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with input and target data – produced by data_utils.split_data",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        required=True,
        help="Directory where to store generated embeddings",
    )
    parser.add_argument(
        "--num-days",
        nargs="*",
        type=int,
        default=[1, 7, 30],
        help="Number of days to compute features",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top column values to consider in feature generation",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="If set, split data is used for embeddings generation. If not set, all data is used.",
    )
    return parser


def main(params):
    data_dir = DataDir(Path(params.data_dir))

    embeddings_dir = Path(params.embeddings_dir)

    split = params.split
    print(split)

    relevant_client_ids = load_relevant_clients_ids(input_dir=data_dir.input_dir)
    client_ids, embeddings = create_embeddings(
        data_dir=data_dir,
        num_days=params.num_days,
        top_n=params.top_n,
        relevant_client_ids=relevant_client_ids,
        split=split
    )

    save_embeddings(
        client_ids=client_ids,
        embeddings=embeddings,
        embeddings_dir=embeddings_dir,
    )


if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main(params=params)
