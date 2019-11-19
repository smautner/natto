
from collections import defaultdict
from natto import hungutil as hu
import numpy as np

from natto.util.copkmeans import cop_kmeans as ckmeans
def cluster(a,b,ca,cb, debug=False,normalize=True,draw=lambda x,y:None,maxsteps=6):
    
    
    ro,co,dists = hu.hungarian(a,b)
    print (len(ro), len(co), dists.shape)
    for i in range(maxsteps):
        constraints = getconstraints(ro,co,dists,ca,cb, draw=draw) 
        cb, _ = ckmeans(b,k=len(np.unique(cb)), ml=constraints,cl=[])
        print("changed b:")
        draw(ca,cb)
        constraints = getconstraints(co,ro,dists,cb,ca, reverse=True, draw=draw) 
        ca, _ = ckmeans(a,k=len(np.unique(ca)), ml=constraints,cl=[])
        print("changed a:")
        draw(ca,cb)

    return ca,cb, None


def getconstraints(ro,co,dists,ca,cb, reverse=False, draw= lambda x,y:None): 
    
    # for each cluster in a: 
    # get all matching cells, drop those with high dist
    
    mustlink = []
    conlist=[]
    di = defaultdict(list)
    for a,b in zip(ro,co):
        di[ca[a]].append( ( dists[a,b] if not reverse else dists[b,a] ,b)  )

    for tlist in di.values():
        tlist.sort(reverse=True)
        cut = int(len(tlist)*.8)
        tlist1 =  [ b for a,b in  tlist[:cut]] 
        arglist =[ b for a,b in  tlist[cut:] ] 
        conlist+=arglist
        mustlink += [(bb,bbb) for bb in tlist1 for bbb in tlist1]

    print("this should highlight the change:")

    z=np.array(cb)
    z[conlist]=-1

    partner = dict( zip(ro,co))
    if not reverse:
        draw( ca ,[ ca[ partner.get(zz,-1) ] if zz != -1 else -1 for zz in z ])
    if reverse:
        draw( z ,ca)




    return mustlink
    






    

