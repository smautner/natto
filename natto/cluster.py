import anndata as ad

import numpy as np
from sklearn import  mixture
import scanpy as sc


def predictgmm(n_classes,X):
    algorithm = mixture.GaussianMixture( n_components=n_classes, covariance_type='full')
    algorithm.fit(X)
    # agglo can not just predict :((( so we have this here
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)
    return y_pred

def predictlou(_,X,params={}):
    adata = ad.AnnData(X)
    sc.pp.neighbors(adata, **params)
    sc.tl.louvain(adata)
    return np.array([int(x) for x in adata.obs['louvain']])

