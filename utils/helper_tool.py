import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

try:
    import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
except ImportError:
    cpp_subsampling = None

try:
    import nearest_neighbors as nearest_neighbors
except ImportError:
    nearest_neighbors = None
    from sklearn.neighbors import NearestNeighbors as SklearnNearestNeighbors


class DataProcessing:
    @staticmethod
    def knn_search(support_pts, query_pts, k):
        if nearest_neighbors is not None:
            neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
            return neighbor_idx.astype(np.int32)

        batch_size = support_pts.shape[0]
        all_neighbors = []
        for b in range(batch_size):
            nbrs = SklearnNearestNeighbors(n_neighbors=k, algorithm='auto').fit(support_pts[b])
            _, indices = nbrs.kneighbors(query_pts[b])
            all_neighbors.append(indices)
        return np.array(all_neighbors, dtype=np.int32)

    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        if cpp_subsampling is not None:
            if (features is None) and (labels is None):
                return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
            if labels is None:
                return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
            if features is None:
                return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
            return cpp_subsampling.compute(
                points, features=features, classes=labels, sampleDl=grid_size, verbose=verbose
            )

        voxel = np.floor(points / grid_size).astype(np.int64)
        _, indices = np.unique(voxel, axis=0, return_index=True)
        indices = np.sort(indices)

        sampled_points = points[indices]
        sampled_features = features[indices] if features is not None else None
        sampled_labels = labels[indices] if labels is not None else None

        if features is not None and labels is not None:
            return sampled_points, sampled_features, sampled_labels
        if features is not None:
            return sampled_points, sampled_features
        if labels is not None:
            return sampled_points, sampled_labels
        return sampled_points
