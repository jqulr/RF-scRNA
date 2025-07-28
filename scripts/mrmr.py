import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from typing import List

# computes mutual information score between feature and target class
def mutual_info(X, y):
    # "auto": True for dense X / False for sparse X
    return mutual_info_classif(X, y, discrete_features="auto")

# Transforms continuous gene expression data into categorical variables (0,1,2) 
# x: the selected gene's expression across all cells in the shape of (n_samples, )
def discretize_feat(x, n_bins=10):
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    return est.fit_transform(x.reshape(-1, 1)).astype(int).ravel()

# computes mutual information score between the single candidate feature against each of the selected features
def redundancy(X: np.ndarray, idx_selected: List[int], idx_remain: int, n_bins: int) -> float:

    mi_total = 0
    x_remain = discretize_feat(X[:, idx_remain], n_bins)

    if not idx_selected:
        return 0
    
    # compute mi for each gene in the selected list of genes against the remaining gene and sum mi scores
    for i in idx_selected:
        x_select = discretize_feat(X[:, i], n_bins)
        mi = mutual_info_classif(x_select, x_remain)

        mi_total += mi

    redundancy_score = mi_total / len(idx_selected)
    return redundancy_score


# run mRMR to selects genes with minimum redundancy and maximum relevance
# 
def run_mrmr():
    # TODO
    pass


# def mrmr(X, y, k=10):
#     """
#     Perform Minimum Redundancy Maximum Relevance feature selection.
    
#     Parameters:
#         X: ndarray (n_samples x n_features)
#         y: array-like (n_samples,) class labels
#         k: number of features to select

#     Returns:
#         List of selected feature indices
#     """
#     n_features = X.shape[1]
#     selected = []
#     candidate_indices = list(range(n_features))

#     # Compute MI between each feature and the class labels (relevance)
#     relevance = compute_mutual_info(X, y)

#     for _ in range(k):
#         best_score = -np.inf
#         best_feature = None
#         for candidate in candidate_indices:
#             if candidate in selected:
#                 continue
#             redundancy = compute_feature_redundancy(X, selected, candidate)
#             score = relevance[candidate] - redundancy
#             if score > best_score:
#                 best_score = score
#                 best_feature = candidate
#         selected.append(best_feature)
#         candidate_indices.remove(best_feature)
#     return selected