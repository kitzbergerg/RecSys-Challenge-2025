import numpy as np
import pandas as pd
import re

from abc import ABC, abstractmethod, abstractproperty
from datetime import timedelta
from typing import List

from custom_pipline.constants import (
    EMBEDDINGS_DTYPE,
)


def raise_err_if_incorrect_form(string_representation_of_vector: str):
    """
    Checks if string_representation_of_vector has the correct form.

    Correct form is a string representing list of ints with arbitrary number of spaces in between.

    Args:
        string_representation_of_vector (str): potential string representation of vector
    """
    m = re.fullmatch(r"\[( *\d* *)*\]", string=string_representation_of_vector)
    if m is None:
        raise ValueError(
            f"{string_representation_of_vector} is incorrect form of string representation of vector – correct form is: '[( *\d* *)*]'"
        )


def parse_to_array(string_representation_of_vector: str) -> np.ndarray:
    """
    Parses string representing vector of integers into array of integers.

    Args:
        string_representation_of_vector (str): string representing vector of ints e.g. '[11 2 3]'
    Returns:
        np.ndarray: array of integers obtained from string representation
    """
    raise_err_if_incorrect_form(
        string_representation_of_vector=string_representation_of_vector
    )
    string_representation_of_vector = string_representation_of_vector.replace(
        "[", ""
    ).replace("]", "")
    return np.array(
        [int(s) for s in string_representation_of_vector.split(" ") if s != ""]
    ).astype(dtype=EMBEDDINGS_DTYPE)


class Calculator(ABC):
    """
    Calculator interface for computing features and storing their size.
    """

    @abstractproperty
    def features_size(self) -> int:
        """
        Calculates features size for calculator.
        """
        pass

    @abstractmethod
    def compute_features(self, events: pd.DataFrame) -> np.ndarray:
        """
        This method computes features for a single collection of events.

        Args:
            events (pd.DataFrame): DataFrame containing the events data.
        Returns:
            np.ndarray: feature vector
        """
        pass


class QueryFeaturesCalculator(Calculator):
    """
    Calculator class for computing query features for a search_query event type.
    The feature vector is the average of all query embeddings from search_query events in user's history.
    """

    def __init__(self, query_column: str, single_query: str):
        """
        Args:
            query_column (str): Name of column containing quantized text embeddings.
            single_query (str): A sample string representation of quantized (integer) text embedding vector.
        """
        self.query_column = query_column
        self.query_size = len(parse_to_array(single_query))

    @property
    def features_size(self) -> int:
        return self.query_size

    def compute_features(self, events: pd.DataFrame) -> np.ndarray:
        quantized_query_representations = np.stack(
            [
                parse_to_array(string_representation_of_vector=v)
                for v in events[self.query_column].values
            ],
            axis=0,
        )
        return quantized_query_representations.mean(axis=0)



class StatsFeaturesCalculator(Calculator):
    """
    Calculator class for computing statistical features for a given event type.
    The feature vector includes the count of occurrences of specified column values within given time windows in user's history.
    Multiple time windows and columns combination can be used to create features.
    """

    def __init__(
        self,
        num_days: List[int],
        max_date: pd.Timestamp,
        columns: List[str],
        unique_values: dict[str, pd.Index],
    ):
        """
        Args:
            num_days (List[int]): List of time windows (in days) for generating features.
            max_date (datetime): The latest event date in the training input data.
            columns (List[str]): Columns to be used for feature generation.
            unique_values (Dict[List]): A dictionary with each key being a column name and
            the corresponding value being a list of selected
            number of top values for that column.
        """
        self._num_days = num_days
        self._max_date = max_date
        self._columns = columns
        self._unique_values = unique_values

    @property
    def features_size(self) -> int:
        return (
            sum((len(self._unique_values[column]) for column in self._columns))
            * len(self._num_days)
            + 1
        )

    def compute_features(self, events: pd.DataFrame) -> np.ndarray:
        features = np.zeros(self.features_size, dtype=EMBEDDINGS_DTYPE)
        features[0] = events.shape[0]
        pointer = 1
        timestamps = events["timestamp"].sort_values()
        for days in self._num_days:
            start_date = self._max_date - timedelta(days=days)
            idx = timestamps.searchsorted(start_date)
            for column in self._columns:
                features_to_write = features[
                    pointer : pointer + len(self._unique_values[column])
                ]
                values = events[column].to_numpy()[idx:]
                for val in np.unique(values):
                    features_to_write[self._unique_values[column] == val] += np.sum(
                        values == val
                    )
                pointer += len(self._unique_values[column])
        return features

class RecencyCalculator(Calculator):
    def __init__(self, max_date: pd.Timestamp):
        self.max_date = max_date

    @property
    def features_size(self) -> int:
        return 1

    def compute_features(self, events: pd.DataFrame) -> np.ndarray:
        if events.empty:
            return np.array([0.0], dtype=EMBEDDINGS_DTYPE)
        last_event = events["timestamp"].max()
        recency_days = (self.max_date - last_event).days
        return np.array([recency_days], dtype=EMBEDDINGS_DTYPE)


class DiversityCalculator(Calculator):
    def __init__(self, column: str):
        self.column = column

    @property
    def features_size(self) -> int:
        return 1

    def compute_features(self, events: pd.DataFrame) -> np.ndarray:
        values = events[self.column].value_counts(normalize=True)
        entropy = -(values * np.log(values + 1e-9)).sum()  # avoid log(0)
        return np.array([entropy], dtype=EMBEDDINGS_DTYPE)


class PriceStatsCalculator(Calculator):
    def __init__(self):
        pass
        #self.product_properties = product_properties.set_index("sku")

    @property
    def features_size(self) -> int:
        return 4

    def compute_features(self, events: pd.DataFrame) -> np.ndarray:
        prices = events["price"].dropna().astype(float)
        if prices.empty:
            return np.zeros(4, dtype=EMBEDDINGS_DTYPE)
        
        #print(f"{prices.mean()=}, {prices.min()=}, {prices.max()=}, {prices.std() if len(events) > 1 else 0.0}")
        return np.array([
            prices.mean(),
            prices.min(),
            prices.max(),
            prices.std() if len(events) > 1 else 0.0
        ], dtype=EMBEDDINGS_DTYPE)


class CartAbandonmentCalculator(Calculator):
    def __init__(self, buy_events: pd.DataFrame):
        self.buy_skus = set(buy_events["sku"])

    @property
    def features_size(self) -> int:
        return 1

    def compute_features(self, add_to_cart_events: pd.DataFrame) -> np.ndarray:
        if add_to_cart_events.empty:
            return np.array([0.0], dtype=EMBEDDINGS_DTYPE)
        added_skus = set(add_to_cart_events["sku"])
        abandoned = added_skus - self.buy_skus
        abandonment_ratio = len(abandoned) / max(len(added_skus), 1)
        return np.array([abandonment_ratio], dtype=EMBEDDINGS_DTYPE)

class RemoveFromCartCalculator(Calculator):
    def __init__(self):
        pass

    @property
    def features_size(self) -> int:
        return 2

    def compute_features(self, events: pd.DataFrame) -> np.ndarray:
        if events.empty:
            return np.zeros(self.features_size, dtype=EMBEDDINGS_DTYPE)

        total_removes = len(events)
        unique_skus = events["sku"].nunique() if "sku" in events.columns else 0

        return np.array([total_removes, unique_skus], dtype=EMBEDDINGS_DTYPE)

class BuyStatsCalculator(Calculator):
    def __init__(self):
        self._features_size = 3  # total_buys, unique_buys, buys_per_day

    @property
    def features_size(self) -> int:
        return self._features_size

    def compute_features(self, events: pd.DataFrame) -> np.ndarray:
        if events.empty:
            return np.zeros(self.features_size, dtype=EMBEDDINGS_DTYPE)

        events["timestamp"] = pd.to_datetime(events["timestamp"])
        total_buys = len(events)
        unique_buys = events["sku"].nunique()
        timespan = (events["timestamp"].max() - events["timestamp"].min()).days + 1
        buys_per_day = total_buys / max(timespan, 1)

        return np.array([total_buys, unique_buys, buys_per_day], dtype=EMBEDDINGS_DTYPE)

#add feature for interaction duration based on timestamp events
class InteractionDurationCalculator(Calculator):
    @property
    def features_size(self) -> int:
        return 1

    def compute_features(self, events: pd.DataFrame) -> np.ndarray:
        if events.empty or "timestamp" not in events.columns:
            return np.array([0.0], dtype=EMBEDDINGS_DTYPE)
        duration = (events["timestamp"].max() - events["timestamp"].min()).total_seconds()/3600
        if duration == np.inf or duration < 0:
            duration = 0.0
        return np.array([duration], dtype=EMBEDDINGS_DTYPE)

    
class DaysDistributionCalculator(Calculator):
    @property
    def features_size(self) -> int:
        return 7

    def compute_features(self, events: pd.DataFrame) -> np.ndarray:
        if events.empty or "timestamp" not in events.columns:
            return np.zeros(7, dtype=EMBEDDINGS_DTYPE)

        weekdays = pd.to_datetime(events["timestamp"]).dt.weekday
        counts = weekdays.value_counts(normalize=True).sort_index()
        distribution = np.zeros(7, dtype=EMBEDDINGS_DTYPE)
        distribution[counts.index.to_numpy()] = counts.to_numpy()
        return distribution

#to find the gaps between events and put them in features
class TimeEventDiffCalculator(Calculator):
    @property
    def features_size(self) -> int:
        return 4  

    def compute_features(self, events: pd.DataFrame) -> np.ndarray:
        if events.empty or "timestamp" not in events.columns:
            return np.zeros(self.features_size, dtype=EMBEDDINGS_DTYPE)

        times = events["timestamp"].sort_values()
        deltas = times.diff().dropna().dt.total_seconds()/3600
        if deltas.empty:
            return np.zeros(self.features_size, dtype=EMBEDDINGS_DTYPE)

        return np.array([
            deltas.mean(),
            deltas.median(),
            deltas.min(),
            deltas.max(),
        ], dtype=EMBEDDINGS_DTYPE)


class MonthDistributionCalculator(Calculator):
    @property
    def features_size(self) -> int:
        return 12  # one column per month (Jan to Dec)

    def compute_features(self, events: pd.DataFrame) -> np.ndarray:
        distribution = np.zeros(12, dtype=EMBEDDINGS_DTYPE)
        if events.empty or "timestamp" not in events.columns:
            return distribution

        months = events["timestamp"].dt.month  # 1 to 12
        counts = months.value_counts(normalize=True).sort_index()
        distribution[counts.index.to_numpy() - 1] = counts.to_numpy()  # subtract 1 for zero-based indexing

        return distribution

class PageVisitCalculator(Calculator):
    @property
    def features_size(self) -> int:
        return 3

    def compute_features(self, events: pd.DataFrame) -> np.ndarray:
        # check only on unique urls in page_visit events
        count = len(events)
        if events.empty or "url" not in events.columns:
            unique_urls = 0.0
            ratio = 0.0
        else:
            unique_urls = events["url"].nunique()
            #revisits
            url_counts = events["url"].value_counts()
            revisits = url_counts[url_counts > 1].sum() - (url_counts > 1).sum()
            total = len(events)
            ratio = revisits / max(total, 1)


        
        return np.array([count, unique_urls, ratio], dtype=EMBEDDINGS_DTYPE)
    
class QueryCountCalculator(Calculator):
    @property
    def features_size(self) -> int:
        return 2

    def compute_features(self, events: pd.DataFrame) -> np.ndarray:
        # check only on unique query in page_visit events
        count = len(events)
        if events.empty or "query" not in events.columns:
            unique_queries = 0.0
            
        else:
            unique_queries = events["query"].nunique()
        
        return np.array([count, unique_queries], dtype=EMBEDDINGS_DTYPE)


class ProductNameFeaturesCalculator(Calculator):
    """
    Calculator class for computing query features for a Product event type referring to the name.
    The feature vector is the average of all query embeddings from search_query events in user's history.
    """

    def __init__(self, name_column: str, single_name: str):
        """
        Args:
            name_column (str): Name of column containing quantized text embeddings.
            single_name (str): A sample string representation of quantized (integer) text embedding vector.
        """
        self.name_column = name_column
        self.name_size = len(parse_to_array(single_name))

    @property
    def features_size(self) -> int:
        return self.name_size

    def compute_features(self, events: pd.DataFrame) -> np.ndarray:
        quantized_name_representations = np.stack(
            [
                parse_to_array(string_representation_of_vector=v)
                for v in events[self.name_column].values
            ],
            axis=0,
        )
        return quantized_name_representations.mean(axis=0)


class CombinedCalculator(Calculator):
    def __init__(self, calculators: List[Calculator]):
        self._calculators = calculators

    @property
    def features_size(self) -> int:
        for calc in self._calculators:
            print(f"Calculator: {type(calc).__name__} + feature_size: {calc.features_size}")
        return sum(calc.features_size for calc in self._calculators)

    def compute_features(self, events: pd.DataFrame) -> np.ndarray:
        features_list = []
        for calc in self._calculators:
            result = calc.compute_features(events)
            if np.isnan(result).any():
                print(f"NaN detected in calculator: {type(calc).__name__}")
            features_list.append(result)
        return np.concatenate(features_list, axis=0)