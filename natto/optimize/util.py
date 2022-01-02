from natto.process import Data
from scipy.sparse import csr_matrix
import time
from natto.out import quality as Q
import natto.process as p
from sklearn.neighbors import NearestNeighbors as NN
import matplotlib
from natto.process import noise
matplotlib.use('Agg')

"""many functions to calculate distances between data sets"""

def rundist(arg):
    loader,nc,scale = arg
    m =  Data().fit(*loader(),
                       debug_ftsel=False,
                       scale = scale,
                       maxgenes=800,
                       quiet=True,
                       pca = 20,
                       titles=("3", "6"),
                       make_even=True)
    clust = lambda ncc : Q.rari_score(*p.gmm_2(*m.dx,nc=ncc,cov='full'), *m.dx)[0]
    return [clust(ncc) for ncc in nc] 



def rundist_2loaders(arg):
    l1,l2,nc = arg

    m =  Data().fit(l1(),l2(),
                    debug_ftsel=False,
                    quiet=True,
                    pca = 20,
                    titles=("3", "6"),
                    make_even=True)
    labels = p.gmm_2(*m.dx,nc=nc,cov='full')
    labels2 = p.gmm_2(*m.dx,nc=10,cov='full')
    a,b =  Q.rari_score(*labels, *m.dx)
    c,d =  Q.rari_score(*labels2, *m.dx)
    return a,c


def normal(arg):
    l1,l2,pca, scale = arg
    m =  Data().fit(l1(),l2(),
                    debug_ftsel=False,
                    quiet=True,
                    scale = scale,
                    pca = pca,
                    dimensions=20,
                    titles=("3", "6"),
                    make_even=True)
    #labels = p.gmm_2(*m.dx,nc=nc,cov='full')
    #a,b =  Q.rari_score(*labels, *m.dx)
    nc = [15]
    return [Q.rari_score(*p.gmm_2(*m.dx,nc=NC,cov='full'), *m.dx)[0] for NC in nc]

def samplenum(arg):
    loader,samp = arg
    t = time.time()
    m =  Data().fit(*loader(seed=None, subsample=samp),
                    debug_ftsel=False,
                    quiet=True,
                    pca = 20,
                    titles=("3", "6"),
                    make_even=True)
    #labels = p.gmm_2(*m.dx,nc=nc,cov='full')
    #a,b =  Q.rari_score(*labels, *m.dx)
    t1 =  time.time() - t
    a = Q.rari_score(*p.gmm_2(*m.dx,nc=15,cov='full'), *m.dx)[0]
    t2 =  time.time() - t
    b = Q.rari_score(*p.gmm_2(*m.dx,nc=15,cov='tied'), *m.dx)[0]
    return a,b, t1,t2




#######################
# BASEKLINE 
########################



def onn(a,b):
    mod = NN(n_neighbors = 1, metric = 'euclidean')
    mod.fit(b)
    _,d  = mod.kneighbors(a)
    return d.mean()
    
def sim(a,b):
    return (onn(a,b)+onn(b,a))/2


def rundist_1nn_2loaders(arg):
    l1,l2,pca,scale = arg

    m =  Data().fit(l1(),l2(),
                    debug_ftsel=False,
                    quiet=True,
                    scale=scale, 
                    pca = pca,
                    titles=("3", "6"),
                    make_even=True)

    return sim(*m.dx), sim(*m._toarray())


def get_noise_run_moar(args):
    loader, cluster, level, metrics = args
    if level == 0:
        return [1]*len(cluster)*len(metrics) # should be as long a a normal return 

    adat = loader()
    if type(adat.X) != csr_matrix:
        adat.X = csr_matrix(adat.X)

    # todo: also loop over cluster algos
    m =  noise.get_noise_single(adat, level)
    r = [Q.rari_score(*c(*m.dx) , *m.dx, metric = metric) for c in cluster for metric in metrics]
    return r


def get_noise_run(args):
    loader, pool, cluster= args
    adat = loader()
    noiserange=  range(0,110,10)
    if type(adat.X) != csr_matrix:
        adat.X = csr_matrix(adat.X)

    # i could use mp in get_noise_data i think
    rdydata= noise.get_noise_data(adat, noiserange, title="magic", poolsize=pool, cluster=cluster)
    r = [ Q.rari_score(*m.labels, *m.dx) for m in rdydata]
    return r
