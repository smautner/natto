from natto.input import load
from natto.optimize import distances  as d
import numpy as np 
from functools import partial
from basics.sgexec import sgeexecuter as sge


k3 = partial(load.load3k6k, subsample=1500,seed=None)
p7 = partial(load.loadp7de, subsample=1500, seed=None)
immune = partial(load.loadimmune, subsample=1500, seed=None)

s = sge()
numclusters=[8,16,25]
for loader in [k3, p7, immune]:
    for nc in numclusters:
        s.add_job( d.rundist , [(loader, nc) for r in range(50)])
rr= s.execute()




print(rr)


def p(level, c):
    l =np.array(level)
    return l.mean(axis = 0 )[c]

def ps(level, c):
    l =np.array(level)
    return l.std(axis = 0 )[c]

ctr=0
for l in ['3k','p7','immune']:
    for c in numclusters:
        cda = rr[ctr]
        print (f"{l}, nc: {c}, rari {p(cda,0)}+-{ps(cda,0)}  ari {p(cda,1)}+={ps(cda,1)}")
        ctr+=1


