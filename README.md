# 📊 FeatureClus: Feature Selection for Clustering Models

Welcome to **FeatureClus**, a Python library designed to simplify **feature selection** for **clustering models**. This tool helps you select the most relevant features that enhance clustering performance, ensuring you avoid the "curse of dimensionality" and make your clustering algorithms more efficient and interpretable. 🧠

## 🔍 How It Works

The feature selection process is driven by evaluating how each feature impacts the clustering results. **FeatureClus** uses an isolated data shift for each feature to assess its importance. The process follows these steps:

1. **MinMaxScaler**: First, we scale the features using MinMaxScaler to normalize the data.
2. **PCA (80% variance)**: Next, we apply Principal Component Analysis (PCA) to reduce dimensionality, retaining 80% of the variance.
3. **DBSCAN Clustering**: After reducing the dimensionality, DBSCAN is used to perform clustering.
4. **Silhouette Score Calculation**: For each feature, we calculate the silhouette score to evaluate the quality of the clusters. The silhouette score represents how similar an object is to its own cluster compared to other clusters.
5. **Data Shift and Feature Importance**: By applying isolated shifts to each feature and recalculating the silhouette score, we measure how the score changes. The absolute difference in the silhouette score after shifting each feature is used to rank the features by importance.
6. **Gower matrix**: In case of dealing with categorical variables, please use gower matrix.

This method ensures that the features are evaluated for their individual contribution to the clustering process, allowing you to focus on the most impactful features.

## 🚀 Key Features
- 🔍 **Feature Ranking**: Ranks features based on the absolute change in silhouette score after applying isolated shifts to each feature.
- 📈 **Cluster Evaluation Metrics**: Calculates the silhouette score to assess the clustering quality and the influence of each feature.
- 💻 **Easy-to-Use API**: A simple, intuitive API that can be easily integrated into your machine learning pipeline.


## 📦 Installation

To install the library, run the following command:

```bash
pip install featclus
```

## 📊 Example

Here is a quick example of how to use **FeatureClus** with a clustering algorithm (e.g., KMeans):

```python
from featureclus.model import FeatureSelection
from sklearn.datasets import make_blobs
import pandas as pd

# Sample DataFrame
data, labels = make_blobs(n_samples=10000, centers=7, n_features=15, random_state=42)
df = pd.DataFrame(data, columns=[f"Feature_{i}" for i in range(15)])

# Initialize the FeatureSelection
model = FeatureSelection(data=df, shifts=[1, 25, 50, 75, 100], n_jobs=-1, use_gower=False)

# See how the metrics are important
metrics = model.get_metrics()

```

## 🛠️ Methods

### `get_metrics()`
Returns metrics that assess how each feature contributes to clustering.

### `plot_results(n_features)`
Selects the top `n_features` features based on their importance to clustering results.


## ☕ Support the Project

If you find this inventory optimization tool helpful and would like to support its continued development, consider buying me a coffee. Your support helps maintain and improve this project!

[![Buy Me A Coffee](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.paypal.com/paypalme/sebassarasti)

### Other Ways to Support
- ⭐ Star this repository
- 🍴 Fork it and contribute
- 📢 Share it with others who might find it useful
- 🐛 Report issues or suggest new features

Your support, in any form, is greatly appreciated! 🙏

### Contributions

1. Open a issue.
2. Create a new branch based for solving this.
3. Send the PR to `dev` to test the inner functionality.

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Happy clustering! 🎉
