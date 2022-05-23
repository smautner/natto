import numpy as np
from ubergauss import tools
from natto import input
from natto.optimize.plot import dendro

def neighs(fname = 'jacc/500.dmp', k = 2, draw = False):
    m  = tools.loadfile(fname)
    names = input.get40names()
    sm = tools.spacemap(names)

    if draw:
        shortnames = [n[:5] for n in names]
        mysm = tools.spacemap(shortnames)
        print(shortnames)
        dendro.drawclustermap(np.median(m, axis =2),mysm.integerlist,mysm.getitem)


    codes = getnn(m,k)
    res = [[sm.getitem[z] for z in a] for a in codes]
    return res

def getnn(m,n):
    m = np.median(m,axis = 2)
    np.fill_diagonal(m, np.NINF)
    srt=  np.argsort(m, axis= 1)
    return [ [i]+srt[i,-n:].tolist() for i in range(srt.shape[0]) ]


if __name__ == "__main__":
    print(neighs())



