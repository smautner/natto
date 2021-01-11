

"""
we repeatedly run the distance thing on 3k  to optimize the parameters of preprocessing
"""
from natto.input import load
from natto.input.preprocessing import Data
from natto.process import hungutil as h
from natto.out import quality as q



from lmz import *
#import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


import sklearn
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import scanpy as sc
import umap
import time
import anndata as ad
import ubergauss as ug


class Data2(Data):

    def mk_pca(self, PCA):
        a,b= self._toarray()
        if PCA:
            ab= np.vstack((a, b))
            pca = sklearn.decomposition.PCA(n_components=PCA)
            ab = StandardScaler().fit_transform(ab)

            pca.fit(ab)
            ab = pca.transform(ab)
            var = pca.explained_variance_ratio_
            v = ug.between_gaussians(var)
            a = ab[:len(a),:v+1]
            b = ab[len(a):,:v+1]

            #a = ab[:len(a)]
            #b = ab[len(a):]

        self.pca = a,b,v


# get data
def getdata(  mindisp= 1.45, maxgenes = False, pca= 30, umapdim =8):
    print("ohelpgetdata:",mindisp,maxgenes,pca,umapdim)
    loader = load.load3k6k
    A,B = loader()
    #B = A.copy()
    subsample = 2000
    sc.pp.subsample(A, fraction=None, n_obs=subsample, random_state=None, copy=False)
    sc.pp.subsample(B, fraction=None, n_obs=subsample, random_state=None, copy=False)

    return Data2().fit( A,B,

	   mindisp=mindisp,
	   maxgenes=maxgenes,
	   pca = pca,
	   dimensions=umapdim,
	   ft_combine = lambda x,y: x or y,
	   umap_n_neighbors=10, # used in example
	   maxmean= 4,
	   minmean = 0.02,
	   corrcoef=False,
	   mitochondria = "mt-",
	   pp='linear',
	   debug_ftsel=False,
	   scale=False,
           quiet = True,
	   titles =  ("3","6"),
	   make_even=True)

### get numberz :)
getnumber = lambda y,x: q.rari_score(*y,*x)[0]

def clustandscore(d):
    t = h.cluster_ab(*d.dx, nc =9, cov = 'tied' )
    f = h.cluster_ab(*d.dx, nc =9, cov = 'full' )
    #print ("asdasd",t,f,d.dx)
    t= getnumber(t,d.dx)
    f= getnumber(f,d.dx)
    #print(f"{t} {f}")
    return t,f


def get_data_mp(x):
    return clustandscore(getdata(*x))



