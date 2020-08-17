

from scipy.sparse import csr_matrix
import random
import copy
import numpy as np
import umap
from natto.input.preprocessing import Data
import basics as ba 

class fromdata: 
    def fit(self, things):
        self.things =  list(things)
    def get(self): 
        return random.choice(self.things) 

def noisiate(mtx,noise_percentage,noisemodel= fromdata()): 
    
    # column wise allply noise 
    noisy = mtx.copy().transpose() 
    for row in range(noisy.shape[0]):
        #distribution of values
        rval = noisy[row,:].todense().A1
        noisemodel.fit(rval)
        # write new numbers to rval 
        loc = ( np.random.rand(noisy.shape[1]) < noise_percentage ).nonzero()[0]
        values  = np.array([noisemodel.get() for r in range(len(loc))])
        rval[loc] = values
        # somehow integrate this in noisy... 
        noisy[row] = csr_matrix(rval) # do it like this to prevent explicit 0s everywhere
    return noisy.transpose()
  
def noisiate_adata(adata,noise_percentage=.05, noisemodel=fromdata()): 
    
    # OK THIS IS HOW WE NOISE
    # 1. have a global percentage of affected cells
    # 2. pick from distri (with zeros) 
    #matrix to fill with noise 
    adata.X = noisiate(adata.X,noise_percentage, noisemodel )
    return adata


#############
# NOISE can be generated before or after selecting genes....
############
def appnoize(m,level):
    print("mixing..")
    #re = copy.deepcopy(m)
    re = m 
    if level > 0: 
        re.b.X = noisiate(re.b.X,level/100)
        re.mk_pca(pca)
        re.dx = re.umapify(6,10)
        re.d2 = re.umapify(2,10)
    else: 
        re.mk_pca(pca)
        us = umap.UMAP(n_components = 2, n_neighbors=10, random_state=24).fit(re.pca[0])
        ul = umap.UMAP(n_components = 8, n_neighbors=10, random_state=24).fit(re.pca[0])
        re.dx =    [ul.transform(re.pca[0])]*2
        re.d2 =    [us.transform(re.pca[0])]*2
    re.titles = (f"{title}", f"{title} {level}% noise")
    return re

def get_noise_data(adat, noiserange,title):
    bdat  = adat.copy()
    pca =30
    m = Data().fit(adat, bdat,
                               mindisp=1.0,
                               maxgenes=False,
                               ft_combine = lambda x,y: x or y,
                               corrcoef=False,
                               minmean = 0.02,
                               mitochondria = "mt-",
                               maxmean= 4,
                               pp='linear',
                               pca = pca,
                               dimensions=8,
                               umap_n_neighbors=10, # used in example
                               debug_ftsel=True,
                               scale=False,
                               make_even=True) 


    rdydata=ba.mpmap(appnoize,[(m,noise) for noise in noiserange ]  )#   [ appnoize(m,noise) for noise in noiserange]
    return rdydata

#############
# slownoise 
############


def get_noise_data_slow(loader, noiserange,title):
    numcells = 1000

    def fit(percnoise,loader, tits):
        m = Data().fit(loader(subsample=numcells),
			       noisiate_adata(loader(subsample=numcells),noise_percentage=noise/100),
			       mindisp=0.3,
			       maxgenes=False,
			       ft_combine = lambda x,y: x and y,
			       corrcoef=False,
			       minmean = 0.02,
			       mitochondria = "mt-",
			       maxmean= 4,
			       pp='linear',
			       pca = 30,
			       dimensions=8,
			       umap_n_neighbors=10, # used in example
			       debug_ftsel=True,
			       scale=False,
			       titles = tits, 
			       make_even=True) 
        print("ONE DONE ")
        return m 

    rdydata=[fit(noise,loader, (f"{title}", f"{title} {noise}% noise") ) for noise in noiserange]
    return rdydata

