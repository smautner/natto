import anndata as ad
import numpy as np
import scanpy as sc
import ubergauss as ug


def gmm_2(X1, X2, nc=None, cov ='tied'):
    return gmm_1(X1, nc, cov), gmm_1(X2, nc, cov)


def gmm_1(X, nc=None, cov ='tied'):
    if nc:
        d= {"nclust_min":nc, "nclust_max":nc, "n_init": 40, "covariance_type":cov}
    else:
        d= {"nclust_min":4, "nclust_max":20, "n_init": 20, 'covariance_type':cov}
    return ug.get_model(X, **d).predict(X)


def louvain_1(X, params={}):
    adata = ad.AnnData(X.copy())
    sc.pp.scale(adata, max_value=10)
    sc.pp.neighbors(adata, **params)
    sc.tl.louvain(adata)
    return np.array([int(x) for x in adata.obs['louvain']])


def leiden_2(X, X2, params={}, resolution=1):
    return leiden_1(X, params, resolution), leiden_1(X2, params, resolution)


def leiden_1(X, params={}, resolution = .5):
    adata = ad.AnnData(X.copy())
    sc.pp.scale(adata, max_value=10)
    sc.pp.neighbors(adata, **params)
    sc.tl.leiden(adata, resolution = resolution)
    return np.array([int(x) for x in adata.obs['leiden']])

def coclust(X1,X2,algo=lambda x: leiden_1(x,resolution=.5)):
    X = np.concatenate((X1,X2))
    y= algo(X)
    l1 = X1.shape[0]
    return y[:l1], y[l1:]
