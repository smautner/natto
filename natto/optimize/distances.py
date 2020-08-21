from natto.input.preprocessing import Data
from natto.out import quality as Q
import natto.process as p

def rundist(arg):
    loader,nc = arg

    m =  Data().fit(*loader(),
                       debug_ftsel=False,
                       maxgenes=800,
                       quiet=True,
                       pca = 20,
                       titles=("3", "6"),
                       make_even=True)
    labels = p.gmm_2(*m.dx,nc=nc,cov='full')
    return Q.rari_score(*labels, *m.dx)



def rundist_2loaders(arg):
    l1,l2,nc = arg

    m =  Data().fit(l1(),l2(),
                    debug_ftsel=False,
                    quiet=True,
                    titles=("3", "6"),
                    make_even=True)
    labels = p.gmm_2(*m.dx,nc=nc,cov='full')
    labels2 = p.gmm_2(*m.dx,nc=10,cov='full')
    a,b =  Q.rari_score(*labels, *m.dx)
    c,d =  Q.rari_score(*labels2, *m.dx)
    return a,c
