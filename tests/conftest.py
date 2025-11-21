import pytest
import pandas as pd
from sklearn.datasets import make_blobs


from featclus.model import FeatureSelection


@pytest.fixture
def data_dummy():
    data, labels = make_blobs(n_samples=1000, centers=4, n_features=15, random_state=42)
    columns = [f"feature_{i}" for i in range(15)]
    df = pd.DataFrame(data, columns=columns)
    return df


@pytest.fixture
def data_results(data_dummy):
    df = data_dummy
    model = FeatureSelection(data=df)
    return model.get_metrics()
