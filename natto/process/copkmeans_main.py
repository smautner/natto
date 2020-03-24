
from collections import defaultdict
from natto.process import hungutil as hu
import numpy as np
from natto.process.copkmeans import cop_kmeans as ckmeans
def cluster(a,b,ca,cb, debug=False,normalize=True,draw=lambda x,y:None, maxsteps=6):
    
    ro,co,dists = hu.hungarian(a,b)
    print (len(ro), len(co), dists.shape)
    for i in range(maxsteps):
        constraints = getconstraints(ro,co,dists,ca,cb, draw=draw, debug=debug) 
        cb, _ = ckmeans(b,k=len(np.unique(cb)), ml=constraints,cl=[])
        #draw(ca,cb)
        constraints = getconstraints(co,ro,dists,cb,ca, reverse=True, draw=draw, debug=debug) 
        ca, _ = ckmeans(a,k=len(np.unique(ca)), ml=constraints,cl=[])
        #draw(ca,cb)

    return ca,cb, None


def getconstraints(ro,co,dists,ca,cb, reverse=False, draw= lambda x,y:None, debug = False): 
    
    # for each old in a:
    # get all matching cells, drop those with high dist
    
    mustlink = []
    conlist=[]
    di = defaultdict(list)
    for a,b in zip(ro,co):
        di[ca[a]].append( ( dists[a,b] if not reverse else dists[b,a] ,b)  )

    for tlist in di.values():
        tlist.sort(reverse=False)
        cut = int(len(tlist)*.5)
        tlist1 =  [ b for a,b in  tlist[:cut]] 
        arglist =[ b for a,b in  tlist[cut:] ] 
        conlist+=arglist
        mustlink += [(bb,bbb) for bb in tlist1 for bbb in tlist1]

    if debug:
        print("this should highlight the change:")
        z=np.array(co) # ids in b 
        partner = dict( zip(co,ro)) # id_b -> id_a 
        ignored = {z:1 for z in conlist}
        classarray = np.ones(len(cb))*-1 
        print(len(ca), len(classarray))
        for zz in z: 
            if zz not in ignored:
                classarray[zz] = ca[partner.get(zz,-1)] 

        if not reverse:
            draw( ca ,classarray)
        if reverse:
            draw( classarray ,ca)
    return mustlink
    






    

