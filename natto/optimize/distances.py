from natto.input.preprocessing import Data
from natto.out import quality as Q
import natto.process as p
from sklearn.neighbors import NearestNeighbors as NN



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
    m =  Data().fit(*loader(seed=None, subsample=samp),
                    debug_ftsel=False,
                    quiet=True,
                    pca = 20,
                    titles=("3", "6"),
                    make_even=True)
    #labels = p.gmm_2(*m.dx,nc=nc,cov='full')
    #a,b =  Q.rari_score(*labels, *m.dx)
    return Q.rari_score(*p.gmm_2(*m.dx,nc=15,cov='full'), *m.dx)[0], Q.rari_score(*p.gmm_2(*m.dx,nc=15,cov='tied'), *m.dx)[0]



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


