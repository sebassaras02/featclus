import plotly.express as px
import pandas as pd


class ImportancePlotter:
    """
    This class is responsible for plotting feature importance results.
    """

    def __init__(self):
        pass

    def plot_metrics(self, df: pd.DataFrame, n_features=None):
        """
        Plots the feature importance results.

        Args:
            n_features (int, optional): Number of top features to plot. Defaults to None.

        Returns:
            None: Displays a Plotly bar chart.
        """
        if n_features:
            df = df[:n_features]
        fig = px.bar(
            df,
            y="Importance",
            labels={"Importance": "Importance Score", "index": "Features tested"},
            title="Feature Importance Plot",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig.update_traces(marker_line_color="black", marker_line_width=1.5)
        fig.update_traces(hovertemplate="<b>%{y:.4f}</b><extra></extra>")
        fig.show()
