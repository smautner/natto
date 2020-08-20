from natto.input import load
from natto.optimize import distances  as d
import numpy as np 
from functools import partial
from basics.sgexec import sgeexecuter as sge


k3 = partial(load.load3k6k, subsample=1500,seed=None)
p7 = partial(load.loadp7de, subsample=1500, seed=None)
immune = partial(load.loadimmune, subsample=1500, seed=None)
numclusters=list(range(5,100,5))
loaders = [k3,p7,immune]

s = sge()
for loader in loaders:
    for nc in numclusters:
        s.add_job( d.rundist , [(loader, nc) for r in range(50)])
rr= s.execute()



def p(level, c):
    l =np.array(level)
    return l.mean(axis = 0 )[c]

def ps(level, c):
    l =np.array(level)
    return l.std(axis = 0 )[c]

ctr=0
res = np.zeros((2*len(loaders),len(numclusters))) 
for i,l in enumerate( ['3k','p7','immune'] ):
    for j,c in enumerate(numclusters):
        cda = rr[ctr]
        v,st = p(cda,0),ps(cda,0)
        o=i*2
        res[o,j] = v
        res[o+1,j] = st
        ctr+=1

print (res)


