import numpy as np
import pandas as pd

from typing import List
from copy import deepcopy

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

from joblib import Parallel, delayed

import plotly.express as px
import gower

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
        data: pd.DataFrame
            The data to be used in the model.
        shifts: List
            The shifts to be used in the data.

    Returns:
        pd.DataFrame: A DataFrame with the importance of each feature sorted.
    """

    def __init__(self, data: pd.DataFrame, shifts: List = [5, 10, 50], n_jobs: int = 1, model: Pipeline = Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                ("pca", PCA(0.8)),
                ("clustering", DBSCAN()),
            ]),
            use_gower: bool = False):
        self.data = data
        self.shifts = shifts
        self.model = model
        self.use_gower = use_gower
        self.columns = data.columns
        self.n_jobs = n_jobs
        self.cache_history = 0
        self.results = None

    def _shift_data_sc(
        self, df: pd.DataFrame, target_column: str
    ) -> List[pd.DataFrame]:
        """
        This function creates different dataframes based on different shifts.
        """
        data_shifted = []
        for value in self.shifts:
            df1 = deepcopy(df)
            df1[target_column] = df1[target_column].shift(value)
            df1 = df1.dropna()
            data_shifted.append(df1)
        return data_shifted

    def _shift_data_mc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This function creates different dataframes based on different shifts for multi-threading.
        """
        data_shifted = []
        for col in self.columns:
            for value in self.shifts:
                df1 = deepcopy(self.data)
                df1[col] = df1[col].shift(value)
                df1 = df1.dropna()
                data_shifted.append(
                    (col, df1)
                )  # Almacena la columna junto con el DataFrame
        return data_shifted

    def _train_model(self) -> dict:
        """
        This function trains a model for each column with a different data shift.
        """
        if self.n_jobs == 1:
            scores = {}
            for col in self.columns:
                df_to_test = self._shift_data_sc(df=self.data, target_column=col)
                values = []
                for df in df_to_test:
                    values.append(self._get_score(df=df))
                scores[col] = np.mean(values)
            return scores
        else:
            dataframes = self._shift_data_mc(df=self.data)
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._get_score)(df) for _, df in dataframes
            )
            scores = {col: [] for col in self.columns}
            for (col, _), score in zip(dataframes, results):
                scores[col].append(score)
            return {col: np.mean(scores[col]) for col in scores}

    def _process_columns(self, col):
        """
        This function processes the columns of the data.
        """
        df_to_test = self._shift_data(df=self.data, target_column=col)
        values = []
        for df in df_to_test:
            values.append(self._get_score(df=df))
        return col, np.mean(values)

    def _get_score(self, df) -> float:
        """
        This function calculates the silhouete score for each model created.
        """
        if self.use_gower:
            labels = self.model.fit_predict(gower.gower_matrix(df.astype('float64')))
        else:
            labels = self.model.fit_predict(df)
        score = silhouette_score(X=df, labels=labels)
        return score

    def get_metrics(self) -> pd.DataFrame:
        """
        This function saves the results of the metrics and sorts the results.
        """
        scores_shifted = self._train_model()
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

    def plot_results(self, n_features=None):
        """
        This function plots the results of the model.
        """
        if self.cache_history == 0:
            self.get_metrics()
        data = self.results
        if n_features:
            data = data[:n_features]
        fig = px.bar(
            data,
            y="Importance",
            labels={"Importance": "Importance Score", "index": "Features tested"},
            title="Feature Importance Plot",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig.update_traces(marker_line_color="black", marker_line_width=1.5)
        fig.update_traces(hovertemplate="<b>%{y:.4f}</b><extra></extra>")
        fig.show()
