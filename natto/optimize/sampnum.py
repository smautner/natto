from natto.input import load
from natto.optimize import distances  as d
import numpy as np 
from functools import partial
from basics.sgexec import sgeexecuter as sge




numcells=list(range(500,1600,100))
loaders = [load.loadp7de, load.loadimmune]
reps = 25




s = sge()
for loader in loaders:
    for nc in numcells:
        s.add_job( d.samplenum, [(loader, nc) for r in range(reps)])
rr= s.execute()

s.save("sampnum")



def p(level, c):
    l =np.array(level)
    return l.mean(axis = 0 )[c]

def ps(level, c):
    l =np.array(level)
    return l.std(axis = 0 )[c]


ctr=0
for i,l in enumerate( ['p7','immune'] ):
    print ("dataset:", l)
    dist, stds = [],[]
    for j,c in enumerate(numcells):
        cda = rr[ctr]
        v,st = p(cda,0),ps(cda,0)
        ctr+=1
        dist.append(v)
        stds.append(st)
    print ('dist: ',dist)
    print ("std: ",stds)



print('this should be tied cov:')
ctr=0
for i,l in enumerate( ['p7','immune'] ):
    print ("dataset:", l)
    dist, stds = [],[]
    for j,c in enumerate(numcells):
        cda = rr[ctr]
        v,st = p(cda,1),ps(cda,1)
        ctr+=1
        dist.append(v)
        stds.append(st)
    print (dist)
    print (stds)


