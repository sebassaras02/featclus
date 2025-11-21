from featclus.model import FeatureSelection
from sklearn.pipeline import Pipeline

import pandas as pd


class TestFeatureSelection:
    """
    This class contains unit tests for the FeatureSelection class.
    """

    def test_model_initialization(self):
        """
        Test if the arguments when creating a model instance are defined as expected.
        """
        model = FeatureSelection(
            data=pd.DataFrame(), shifts=[1, 2], n_jobs=2, use_gower=True
        )

        assert isinstance(model.data, pd.DataFrame)
        assert isinstance(model.shifts, list)
        assert isinstance(model.model, Pipeline)
        assert isinstance(model.use_gower, bool)
        assert isinstance(model.n_jobs, int)
        assert model.cache_history == 0
        assert not model.results

    def test_data_shift(self, data_dummy):
        """
        This test is created to test the functionality of shifting a column in pandas.
        """
        model = FeatureSelection(data=data_dummy, shifts=[5])
        shifted_data = model._shift_data_sc(df=data_dummy, target_column="feature_0")[0]
        assert len(shifted_data) == len(data_dummy) - 5

    def test_shift_df_columns(self, data_dummy):
        """
        This test is created to test the functionality of shifting all columns in a dataframe.
        """
        model = FeatureSelection(data=data_dummy, shifts=[5])
        result = model._shift_data_mc()
        expected_number_of_shifts = len(data_dummy.columns) * len(model.shifts)
        assert len(result) == expected_number_of_shifts

    def test_scoring_normal(self, data_dummy):
        """
        This test is created to test the scoring functionality of the model.
        """
        model = FeatureSelection(data=data_dummy, shifts=[5])
        score = model._get_score(df=data_dummy)
        assert isinstance(score, float)

    def test_scoring_gower(self, data_dummy):
        """
        This test is created to test the scoring functionality of the model using Gower distance.
        """
        model = FeatureSelection(data=data_dummy, shifts=[5], use_gower=True)
        score = model._get_score(df=data_dummy)
        assert isinstance(score, float)

    def test_get_metrics(self, data_dummy):
        """
        This test is created to test the feature permutation over the different shifts.
        """
        model = FeatureSelection(data=data_dummy, shifts=[5], n_jobs=1)
        results = model.get_metrics()
        assert isinstance(results, pd.DataFrame)

    def test_plot_metrics(self, data_dummy):
        """
        This test is created to test the plotting functionality of the model.
        """
        model = FeatureSelection(data=data_dummy, shifts=[5], n_jobs=1)
        model.get_metrics()
        fig = model.plot_results(5)
        assert fig is None

    def test_model_single_core(self, data_dummy):
        """
        This test is created to test the full model functionality using a single core.
        """
        model = FeatureSelection(data=data_dummy, shifts=[5, 10], n_jobs=1)
        results = model.get_metrics()
        assert isinstance(results, pd.DataFrame)

    def test_model_multi_core(self, data_dummy):
        """
        This test is created to test the full model functionality using multiple cores.
        """
        model = FeatureSelection(data=data_dummy, shifts=[5, 10], n_jobs=2)
        results = model.get_metrics()
        assert isinstance(results, pd.DataFrame)
