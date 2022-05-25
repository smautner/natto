from natto.process import Data
import natto.process.util as hutil
import numpy as np
from scipy.sparse import csr_matrix
import time
from natto.out import quality as Q
import natto.process as p
from sklearn.neighbors import NearestNeighbors as NN
import matplotlib
from natto.process import noise
matplotlib.use('Agg')
from natto import cluster
from ubergauss import tools

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


def natto_distance(data,pid = -2):
    #return Q.rari_score(*cluster.gmm_2(*data.projections[pid],nc=15,cov='full'), *data.projections[pid])
    return Q.rari_score(*cluster.gmm_2(*data.projections[pid],cov='tied',n_init=20), *data.projections[pid])





def clusterAndRari(data,pid = -2, cluster=None,clusterArgs={}):
    labels = [cluster(**clusterArgs).fit_predict(data.projections[pid][i]) for i in range(2)]
    return Q.rari_score(*labels, *data.projections[pid])

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



########################
#
########################
from sklearn.metrics.pairwise import cosine_similarity as cos

def cosine(data, numgenes = 0):
    scr1, scr2 = data.genescores
    if numgenes:
        mask = scr1+scr2
        mask = tools.binarize(mask,numgenes).astype(np.bool)
        scr1 = scr1[mask]
        scr2 = scr2[mask]

    return cos([scr1],[scr2]).item()

def booltopx(ar,num):
    srt = np.sort(ar)
    return np.array(ar) >=srt[-num]

def jaccard(data,ngenes= False):
    # intersect/union
    if not ngenes:
        asd = np.array(data.genes)
    else:
        asd = np.array([ booltopx(d,ngenes)  for d in data.genescores])

    union = np.sum(np.any(asd, axis=0))
    intersect = np.sum(np.sum(asd, axis=0) ==2) # np.sum(np.logicaland(asd[0] , asd[1])
    return intersect/union


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


#####################
# BASELINE, processeddata object
######################
def baseline_raw(data):
    return sim(*data.projections[0])
def baseline_pca(data):
    return sim(*data.projections[1])
def baseline_umap(data):
    return sim(*data.projections[2])

def baseline_hung(data, pid):
    (a,b), dis = hutil.hungarian(*data.projections[pid])
    return np.mean(dis[a,b])



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
