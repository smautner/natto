import numpy as np
from scipy.stats import gmean
import scanpy as sc
import sklearn.neighbors as skn
from scipy.optimize import curve_fit


import anndata as ad
from sklearn import  mixture



########
# Clustering
#########
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



###############
# FEATURE SELECTION
###############
def sc3filter(adata,inplace=True):
    # TODO: need to combine this with min_cnt=2 somehow
    low = adata.X.shape[0]*.06
    high = adata.X.shape[0]*.94
    if inplace:
        sc.pp.filter_genes(adata, max_cells=high, inplace=True)
        sc.pp.filter_genes(adata, min_counts=None, min_cells=low, inplace=True)      
        return adata
    else:
        genes,_ = sc.pp.filter_genes(adata, max_cells=high, inplace=inplace)
        genes2,_=sc.pp.filter_genes(adata, min_counts=None, min_cells=low, inplace=inplace)
        return [a or b for a,b in zip(genes,genes2)]

def soyfilter(adata):
    adatalog = sc.pp.log1p(adata, copy=True)
    select = sc.pp.highly_variable_genes(adatalog, 
                                min_disp=None, max_disp=None, min_mean=None, max_mean=None, n_top_genes=None, 
                                n_bins=20, flavor='seurat', subset=False, inplace=False)
    select  = [a[0] for a in select] 
    return adata[:,select]


# sc3 makes it easy to id high variance genes.. so we do that on top
def sc3plus(adata, inplace=True):
    sc3select = sc3filter(adata,inplace=False)
    adata  = adata.copy()[:,sc3select]
    
    variance = adata.X.todense().var(axis=0).getA1()
    counts = adata.X.todense().sum(axis=0).getA1()
    r,_ = curve_fit(lambda x,a,b,c : abs(a)*(x**2)+abs(b)*x+abs(c),counts,variance)
    z = np.poly1d(np.abs(r))
    selected = [ v >= z(c) for v,c in zip(variance,counts)]

    if not inplace: # return selected genes
        selected.reverse()
        return [ False if not sc else selected.pop() for sc in sc3select ]
    adata = adata[:,selected]
    return adata

def sc3plus2(adata, inplace=True):
    # now we calculate the variance correctly... 
    sc3select = sc3filter(adata,inplace=False)
    adata  = adata.copy()[:,sc3select]
    adense = adata.X.todense()
    counts = np.count_nonzero(adense,axis=0).getA1()
    adense[adense==0]=np.nan
    variance = np.nanvar(adense,axis=0).getA1()
    variance = np.nan_to_num(variance)
    r,_ = curve_fit(lambda x,a,b,c : abs(a)*(x**2)+abs(b)*x+abs(c),counts,variance)
    z = np.poly1d(np.abs(r))
    selected = [ v >= z(c) for v,c in zip(variance,counts)]

    if not inplace: # return selected genes
        selected.reverse()
        return [ False if not sc else selected.pop() for sc in sc3select ]
    adata = adata[:,selected]
    return adata

###############
# LOG TRANSFORM 
###############

def logtransform(adata):
    sc.pp.log1p(adata)
    return adata




###############
# CELL FILTER 
###############

# fabrizios suggestion for outlier detection (more or less)
def cell_graph(adata,neighs=20):
    res = skn.LocalOutlierFactor(n_neighbors=neighs).fit_predict(adata.X)
    res = [x==1 for x in res]
    adata= adata[res,:]
    #adata.obs=adata.obs[res]
    return adata

def cell_reads(adata, cutreads=2):
    # at least one feature muss have > cutreads reads.
    res =adata.X.max(axis=1).todense().A1 >= cutreads
    return adata[res,:]


##########
# NORMALISATION
##########


def CPM(adata):
    # according to the documentation this is called normalize_total #####
    #sc.pp.normalize_total(adata, target_sum=1000000, key_added=None, layers=None, layer_norm=None, inplace=True)
    sc.pp.normalize_per_cell(adata, 100000)
    return adata


def upperquantile(adata):
    divby = [ np.percentile(row.data,75) for row in adata.X] # get the quantiles...
    for i,v in enumerate(divby):
        adata.X.data[adata.X.indptr[i]:adata.X.indptr[i+1]]/=v
    return adata
