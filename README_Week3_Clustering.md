# Week 3 — Unsupervised Learning and Clustering Analysis

Clustering analysis of the public Iris dataset using Python and Scikit-learn.

## Dataset
150 observations, four numerical measurements: sepal length, sepal width, petal length and petal width. Species labels are **not used to fit the clustering model**; they are used only for a post-hoc comparison.

## Methods
- StandardScaler preprocessing
- K-Means (`k=3`, `random_state=42`, `n_init=20`)
- Elbow and silhouette diagnostics
- PCA visualization
- Agglomerative clustering with Ward linkage
- Silhouette, Davies–Bouldin and Calinski–Harabasz metrics

## Files
- `Week_3_Clustering_Analysis_Report.docx` — report
- `week3_clustering_analysis.py` — reproducible Python script
- `requirements_week3.txt` — dependencies
- `iris_clustered_results.csv` — assigned cluster per observation
- `cluster_profile_summary.csv` — feature means by cluster
- `clustering_model_metrics.csv` — model comparison
- PNG charts — visualizations

## Run
```bash
pip install -r requirements_week3.txt
python week3_clustering_analysis.py
```

Dataset docs: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
Original source: https://archive.ics.uci.edu/dataset/53/iris
