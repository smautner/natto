import numpy as np
from ubergauss import tools
from natto import input

def neighs(fname = 'jacc/500.dmp', k = 2):
    m  = tools.loadfile(fname)
    names = input.get40names()
    sm = tools.spacemap(names)
    codes = getnn(m,k)
    res = {sm.getitem[i] : [sm.getitem[z] for z in e]  for i,e in enumerate(codes) }
    return res

def getnn(m,n):
    m = np.median(m,axis = 2)
    np.fill_diagonal(m, np.NINF)
    srt=  np.argsort(m, axis= 1)
    return [ srt[i,-n:].tolist() for i in range(srt.shape[0]) ]


if __name__ == "__main__":
    print(neighs())



