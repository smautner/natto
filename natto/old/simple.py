import numpy as np
import math
from scipy.stats import gmean
import scanpy as sc
import sklearn.neighbors as skn
from scipy.optimize import curve_fit
import anndata as ad
from sklearn import  mixture
import maxdropknee as mdk 
import basics as ba 
import functools as fu
########
# THIS SECTION IS CRAP -> overclust
#########

def get_y(algorithm,X):
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)
    return y_pred

def predictgmm(n_classes,X):
    algorithm = mixture.GaussianMixture( n_components=n_classes, covariance_type='full')
    algorithm.fit(X)
    # agglo can not just predict :((( so we have this here
    return get_y(algorithm,X)

def fitbgmm(n_classes,X):
    return mixture.BayesianGaussianMixture( n_components=n_classes, covariance_type='full').fit(X)


def fitgmm(n_classes=4,X=None):
    return  mixture.GaussianMixture(n_init = 50, init_params='random',
            n_components=n_classes, covariance_type='full').fit(X)

def bic(n_classes=4,X=None):
    return -1* fitgmm(n_classes, X).bic(X)


def predictgmm_mdk(X,cmin=4,cmax=23):
    #getbic = lambda n_comp: fitgmm(n_comp,X).bic(X)
    #values = [getbic(a) for a in range(cmin,cmax)] 
    getbic = fu.partial(bic, X=X)
    values = ba.mpmap(getbic, range(cmin,cmax), chunksize = 1 , poolsize = 4)

    n_clusters =  mdk.maxdropknee(values)+cmin
    print("old/simple,mdropvalues",values)
    return get_y(fitgmm(n_clusters,X)  ,X)

def predictgmm_angle_based(X, n=30, cmin=4, cmax= 20): 
    # this used 
    #http://cs.uef.fi/sipu/pub/BIC-acivs2008
    
    # maybe also consider this:
    # https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html#sphx-glr-auto-examples-mixture-plot-gmm-py
    # THE PAPER DOESNT TALK ABOT THE VALUE FOR N;; also n doesnt matter?

    
    # get all the  values
    getbic = lambda n_comp: fitgmm(n_comp,X).bic(X)
    values = [-getbic(a) for a in range(cmin,cmax)] 
    values.append(values[-1])
    
    # INITIALIZE
    cur,pre,aft = [values[0]]*3

    # GET DIFFs 
    diffs = []
    for i,m in enumerate(range(cmin, cmax)):
        cur = values[i]
        aft = values[i+1]
        diffs.append((i, m, pre+aft-2*cur ))
        pre = cur

    # find local minima in diff function
    locmin = [ v for i,v in enumerate(diffs[:-1]) 
            if diffs[i][2] < diffs[i-1][2] and diffs[i][2] < diffs[i+1][2] ]

    
    # for each n with decreasing order of localmin value ( is this bic or diff?)
    locmin.sort(key= lambda x: x[2], reverse=True) # decreasing order of locmin
    
    cur = -999999999
    lastm = -1
    angle = lambda a,b,c: math.atan(1/abs(b-a)) + math.atan(1/abs(c-b))
    for i,m,v in locmin:
        a = angle(*values[i-1:i+2]) 
        
        if a < cur:
            break
        else:
            cur = a
            lastm = m 
    else:
        print("something went wrong")
    
    # lastm is hte answer 
    # this selects the maximum bic:
    #lastm = np.argmax(values)+cmin
    
    # this is my fuckery 
    '''
    cv=-9999999
    for i,v in enumerate(values):
        if v < cv:
            lastm = i + cmin
            break
        cv=v
        
    # max drop after high 
    cv=-99999999
    drops=[]
    for i,v in enumerate(values):
        if v < cv:
            lastm = i + cmin
            drops.append([v - cv ,lastm])
        cv=v
    drops.sort()
    lastm= drops[0][1]
    print("applying max drop")
    '''
    
    print ("gmm_angles:",lastm, values, [a[2] for a in diffs])
    return get_y(fitgmm(lastm,X)  ,X)

    
        




def predictgmm_BIC(X):
    # GET INITIAL BIC
    algorithm = mixture.GaussianMixture( n_components=3, covariance_type='full')
    algorithm.fit(X)
    ob= algorithm.bic(X)
    old_delta = -1
    # FOR ALL CLUSTERCOUNTS
    for n in range(4,30):
        algorithm = mixture.GaussianMixture( n_components=n, covariance_type='full')
        # GET BIC
        algorithm.fit(X)
        bx = algorithm.bic(X)
        
        # compare delta ch
        # if the new delta is small (50%) .. stop 
        delta = bx-ob
        if  (-old_delta * .5 ) > -delta:
            print("clusters:",n-1, old_delta, delta)
            break
        # save old values
        old_delta = delta
        ob = bx
        oldalgorithm = algorithm 
        
    return get_y(oldalgorithm,X)
        
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
