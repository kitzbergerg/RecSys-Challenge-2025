import argparse
import logging
from pathlib import Path
import pandas as pd

from data_utils.constants import EventTypes
from data_utils.utils import (
    load_with_properties,
)
from data_utils.data_dir import DataDir

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def create_embeddings(data_dir: DataDir):
    product_categories = set()
    urls = set()

    product_buy_df = load_with_properties(data_dir=data_dir, event_type=EventTypes.PRODUCT_BUY)
    product_buy_df["timestamp"] = pd.to_datetime(product_buy_df.timestamp)
    product_name_embedding = product_buy_df["name"].iloc[0]
    product_categories.update(product_buy_df["category"].tolist())

    add_to_cart_df = load_with_properties(data_dir=data_dir, event_type=EventTypes.ADD_TO_CART)
    add_to_cart_df["timestamp"] = pd.to_datetime(add_to_cart_df.timestamp)
    product_categories.update(add_to_cart_df["category"].tolist())

    remove_from_cart_df = load_with_properties(data_dir=data_dir, event_type=EventTypes.REMOVE_FROM_CART)
    remove_from_cart_df["timestamp"] = pd.to_datetime(remove_from_cart_df.timestamp)
    product_categories.update(remove_from_cart_df["category"].tolist())

    page_visit_df = load_with_properties(data_dir=data_dir, event_type=EventTypes.PAGE_VISIT)
    page_visit_df["timestamp"] = pd.to_datetime(page_visit_df.timestamp)
    urls.update(page_visit_df["url"].tolist())

    search_query_df = load_with_properties(data_dir=data_dir, event_type=EventTypes.SEARCH_QUERY)
    search_query_df["timestamp"] = pd.to_datetime(search_query_df.timestamp)
    query_embedding = search_query_df["query"].iloc[0]

    print(f"number of unique urls: {len(urls)}")
    print(f"number of unique product categories: {len(product_categories)}")
    print(f"product name embedding length: {len(product_name_embedding)}, embeddings: {product_name_embedding}")
    print(f"search query embedding length: {len(query_embedding)}, embeddings: {query_embedding}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with input and target data – produced by data_utils.split_data",
    )
    return parser


def main(params):
    data_dir = DataDir(Path(params.data_dir))
    create_embeddings(data_dir=data_dir)


if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main(params=params)
