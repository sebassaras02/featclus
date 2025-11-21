import numpy as np
import pandas as pd

from typing import List, Dict

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

import gower

from .data_processing import DataShifter
from .plots import ImportancePlotter
from .parallelizer import FunctionParallizer


class FeatureSelection:
    """
    This library perforns feature selection for clustering problems.
    The main idea is to shift the data and calculate the silhouette score for each feature.
    The components used are:
        - MinMaxScaler
        - PCA
        - DBSCAN
    The performance of the model is calculated by the silhouette score.

    Args:
        data (pd.DataFrame): The data to be used in the model.
        shifts (List): The shifts to be used in the data.
        n_jobs (int): Number of parallel jobs to run. Default is 1.
        model (Pipeline): The clustering pipeline used for scoring.
        use_gower (bool): Whether to use Gower distance as input for DBSCAN.

    Returns:
        pd.DataFrame: A DataFrame with the importance of each feature sorted.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        shifts: List = [5, 10, 50],
        n_jobs: int = -1,
        model: Pipeline = Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                ("pca", PCA(0.8)),
                ("clustering", DBSCAN()),
            ]
        ),
        use_gower: bool = False,
    ):
        self.data = data
        self.shifts = shifts
        self.model = model
        self.use_gower = use_gower
        self.columns = data.columns
        self.n_jobs = n_jobs
        self.cache_history = 0
        self.results = None

        self.data_shifter = DataShifter()
        self.plotter = ImportancePlotter()
        self.parallelizer = FunctionParallizer(n_jobs=n_jobs)

    def _get_permutation_scores(
        self, df: pd.DataFrame, columns: List, shifts: List
    ) -> Dict:
        """
        Calculates the silhouette scores for multiple shifted columns in parallel.

        Args:
            df (pd.DataFrame): Input dataframe.
            columns (List): List of columns to shift.
            shifts (List): List of shifts to apply.

        Returns:
            dict: Dictionary mapping each column to its average silhouette score.
        """
        dataframes = self.data_shifter.shift_multiple_columns(
            df=df, columns=columns, shifts=shifts
        )
        results = self.parallelizer.handle(
            func=self._get_score,
            iterable=[df for _, df in dataframes],
        )
        scores = {col: [] for col in self.columns}
        for (col, _), score in zip(dataframes, results):
            scores[col].append(score)
        return {col: np.mean(scores[col]) for col in scores}

    def _get_score(self, df) -> float:
        """
        Calculates the silhouette score for the given dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            float: Silhouette score for the clustering result.
        """
        if self.use_gower:
            labels = self.model.fit_predict(gower.gower_matrix(df.astype("float64")))
        else:
            labels = self.model.fit_predict(df)
        score = silhouette_score(X=df, labels=labels)
        return score

    def get_metrics(self) -> pd.DataFrame:
        """
        Computes feature importance based on the difference between
        shifted and original silhouette scores.

        Returns:
            pd.DataFrame: Sorted DataFrame of feature importance scores.
        """
        scores_shifted = self._get_permutation_scores(
            df=self.data, columns=self.columns, shifts=self.shifts
        )
        original_score = self._get_score(df=self.data)
        final_values = {}
        for key, value in scores_shifted.items():
            final_values[key] = np.abs(original_score - value)
        df = pd.DataFrame(
            final_values.values(), columns=["Importance"], index=final_values.keys()
        ).sort_values("Importance", ascending=False)
        self.cache_history = 1
        self.results = df
        return df

    def plot_results(self, n_features: int = 3):
        """
        Plots the feature importance metrics.

        Args:
            n_features (int): Number of top features to plot. Default is 3.
        """
        if self.cache_history == 0:
            self.get_metrics()
        self.plotter.plot_metrics(df=self.results, n_features=n_features)
