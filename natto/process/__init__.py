import anndata as ad
import numpy as np
import scanpy as sc
import ubergauss as ug
from lmz import *
import random
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN, MeanShift, AffinityPropagation, Birch

def gmm_2(X1, X2, nc=None, cov ='full'):
    return gmm_1(X1, nc, cov), gmm_1(X2, nc, cov)


def gmm_1(X, nc=None, cov ='full'):
    if nc:
        d= {"nclust_min":nc, "nclust_max":nc, "n_init": 40, "covariance_type":cov}
    else:
        d= {"nclust_min":4, "nclust_max":20, "n_init": 20, 'covariance_type':cov}
    return ug.get_model(X, **d).predict(X)


def gmm_2_dynamic(X1,X2,nc=(4,20),pool=1):
    
    def getmodels(X):
        train = ug.functools.partial(ug.traingmm,X=X,n_init=30)
        if pool > 1 :
            models = ug.mpmap( train , range(*nc), poolsize= pool)
        else:
            models =[ train(x) for x in range(*nc)]
        return models 

    models = list(zip(getmodels(X1), getmodels(X2)))
    AIC = [m.aic(X1) + m2.aic(X2) for m,m2 in models ]
    print(AIC)
    mod_id = ug.diag_maxdist(AIC)
    a,b =  models[mod_id]
    print(f"dynamic clustering wants: {nc[0]+mod_id} clusters")
    
    return a.predict(X1), b.predict(X2)



def louvain_1(X, params={}):
    adata = ad.AnnData(X.copy())
    sc.pp.scale(adata, max_value=10)
    sc.pp.neighbors(adata, **params)
    sc.tl.louvain(adata)
    return np.array([int(x) for x in adata.obs['louvain']])






def random_assign(X, nc=15):
    numcells = X.shape[0]
    return np.array(random.choices(Range(nc),k=numcells))

def random_2(a,b,nc = 15):
    return random_assign(a,nc), random_assign(b,nc)





def spec_1(X,nc=15): 
     return SpectralClustering(n_clusters=nc).fit_predict(X)

def spec_2(a,b, nc =15): 
    return spec_1(a,nc), spec_1(b,nc)

def kmeans_1(a, nc): 
    return KMeans(n_clusters=nc).fit_predict(a)
def kmeans_2(a,b,nc=15):
    return kmeans_1(a,nc), kmeans_1(b,nc)

def birch_1(a, nc): 
    return Birch(n_clusters=nc).fit_predict(a)
def birch_2(a,b,nc=15):
    return birch_1(a,nc), birch_1(b,nc)

def afprop_1(a, kwargs): 
    return AffinityPropagation(**kwargs).fit_predict(a)
def afprop_2(a,b, **kwargs):
    return afprop_1(a,kwargs), afprop_1(b,kwargs)

def dbscan_1(a, kwargs): 
    return DBSCAN(**kwargs).fit_predict(a)
def dbscan_2(a,b,**kwargs):
    return dbscan_1(a, kwargs), dbscan_1(b,kwargs)

def meansh_1(a, kwargs): 
    return MeanShift(**kwargs).fit_predict(a)
def meansh_2(a,b,**kwargs):
    return meansh_1(a,kwargs), meansh_1(b,kwargs)





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
