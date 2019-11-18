
from natto import hungutil as hu
import numpy as np

from natto.util.copkmeans import cop_kmeans as ckmeans
def cluster(a,b,ca,cb, debug=False,normalize=True,maxerror=.10,maxsteps=6):
    
    
    ro,co,dists = hu.hungarian(a,b)
    
    for i in range(maxsteps):
        constraints = getconstraints(ro,co,dists,ca,cb) 
        clusters, _ = ckmeans(b,k=len(np.unique(cb)), ml=constraints,cl=None)
        print (clusters)
        return
        # TODO VISUALIZE 


def getconstraints(ro,co,dists,ca,cb): 
    
    # for each cluster in a: 
    # get all matching cells, drop those with high dist
    
    mustlink = []
    
    for cid in np.unique(ca): 
        ahit= ro[ca == cid]
        bhit= co[ca == cid]
        di  = dists[ahit,bhit] 
        di.sort()
        cutoff = di[len(di)/2]
        const = [b for a,b in zip(ahit,bhit) if dists[a,b] < cutoff]
        mustlink += [(bb,bbb) for bb in b for bbb in b]

    return mustlink
    






    

