from lmz import Map,Zip,Filter,Grouper,Range,Transpose, grouper
import numpy as np
from ubergauss import tools as ut
from sklearn.metrics.pairwise import cosine_similarity as cos

def cosine(a,b, numgenes = 0):
    scr1, scr2 = a.varm['scores'], b.varm['scores']
    if numgenes:
        mask = scr1+scr2
        mask = ut.binarize(mask,numgenes).astype(np.bool)
        scr1 = scr1[mask]
        scr2 = scr2[mask]
    return cos([scr1],[scr2]).item()

def apply_measures(method, instances, repeats = 5):
    l = len(instances)
    res = np.zeros((l,l,repeats))
    for i,obj_i in enumerate(instances):
        for j,obj_j in enumerate(instances[i:]):
            r = [method(obj_i,obj_j,x) for x in range(repeats)]
            for x,val in enumerate(r):
                res[i,j,x] = val
                res[j,i,x] = val
    return res




def apply_measures_mp(method, instances, repeats = 5):
    # pool.maxtasksperchild
    l = len(instances)
    res = np.zeros((l,l,repeats))


    def func(stuff):
        a,b,seeds,i,j = stuff
        r = [ method(a,b,seed)  for seed in seeds ]
        return r,i,j

    tmptasks =([instances[i],instances[j],list(range(repeats)), i,j]
                for i in range(l) for j in range(i,l))

    for r,i,j in ut.xmap(func,tmptasks):
        for x,val in enumerate(r):
                res[i,j,x] = val
                res[j,i,x] = val

    return res



