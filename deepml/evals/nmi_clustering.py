import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score


def nmi_clustering(features, labels, n_cluster=None):
    """Compute the Normalized Mutual Information between two clusterings.

    Args:
        features (np.array): The input features
        labels (np.array): The input labels
        n_cluster (int, optional): Defaults to None. The number of clusters.

    Returns:
        int: The NMI value.
    """

    n_cluster = n_cluster or len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_cluster, n_jobs=-
                    1, random_state=0).fit(features)
    nmi = normalized_mutual_info_score(
        labels.flatten(),
        kmeans.labels_,
        average_method='arithmetic'
    )
    return nmi
