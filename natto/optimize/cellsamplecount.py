from natto.input import load
from natto.optimize import util  as d

import numpy as np 
from functools import partial
from basics.sgexec import sgeexecuter as sge


"""increasing sample count"""

debug = False

numcells=list(range(500,1100 if debug else 2100,500))
loaders = [load.loadp7de, load.loadimmune]
reps = 2 if debug else 5
#s = sge()
rr= []
for loader in loaders:
    for nc in numcells:
        #s.add_job( d.samplenum, [(loader, nc) for r in range(reps)])
        rr.append([d.samplenum((loader, nc)) for r in range(reps)])
        print ('.',end='')
#rr= s.execute()
#s.save("sampnum")



def p(level, c):
    l =np.array(level)
    return l.mean(axis = 0 )[c]

def ps(level, c):
    l =np.array(level)
    return l.std(axis = 0 )[c]




def getcurve(idx):
    ctr=0
    for i,l in enumerate( ['p7','immune'] ):
        print ("dataset:", l)
        dist, stds = [],[]
        for j,c in enumerate(numcells):
            cda = rr[ctr]
            v,st = p(cda,idx),ps(cda,idx)
            ctr+=1
            dist.append(v)
            stds.append(st)
        print ('dist: ',dist)
        print ("std: ",stds)

print("     A")
getcurve(0)
print("     B")
getcurve(1)
print("     T1")
getcurve(2)
print("     T2")
getcurve(3)
