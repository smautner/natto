from collections import Counter
from sklearn.model_selection import cross_val_predict
from  sklearn.neighbors import KNeighborsClassifier as KNC
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors as NN 
from sklearn.metrics import adjusted_rand_score as rand
# QUALITY MEASSUREMENTS
import math


# e^-dd / avg 1nn dist 

def gausssim(a,b, ca, cb): 
    ncb=np.unique(cb)
    res = {}
    for aa in np.unique(ca):
        if aa in ncb:
            if sum(ca==aa) < 5 or sum(cb==aa)<5:
                res[aa]="NA"
                continue
            avg1= nndist(a,ca,cb,c = aa)
            avg2= nndist(b,cb,ca, c=aa)
            d= supernndist(a,b,ca,cb,c=aa)
            avg= (avg1+avg2)/2
            #print (aa,d,avg1,avg2)
            res[aa]= math.exp(-(d*d) / (avg*avg))
    res = {k:v for k,v in res.items() if v!="NA"}
    return res


def nndist(m,cla,clb,c=None): 
            instances = m[cla==c]
            neighs = NN(n_neighbors=2).fit(instances)
            distances, _  = neighs.kneighbors(instances)
            return np.mean(distances[:,1])

def supernndist(a,b,ca,cb,c=None):
            instances = a[ca==c]
            neighs = NN(n_neighbors=1).fit(instances)
            distances, _  = neighs.kneighbors(b[cb==c])
            return np.mean(distances)
        
    
def compare_heatmap(y11,y12,y21,y22,mata,matb):
    
    from lapsolver import solve_dense
    from sklearn.metrics.pairwise import euclidean_distances as ed
    from natto.hungutil import make_canvas_and_spacemaps
    from natto.util.draw import quickdoubleheatmap
    distances = ed(mata,matb)
    hungmatch = solve_dense(distances)


    def prephtmap(y1,y2):
        # returns: canvas,y1map2,y2map2,row,col
        a,b,c = make_canvas_and_spacemaps(y1,y2,hungmatch,normalize=False)
        d,e  = solve_dense(c)
        return c,a,b,d,e

    comp1 = prephtmap(y11,y12)
    comp2 = prephtmap(y21,y22)
    
    quickdoubleheatmap(comp1,comp2)

    def calcmissmatches(stuff):
        canv = stuff[0] 
        r,c = stuff[-2:]
        for rr,cc in zip(r,c):
            canv[rr,cc]=0
        return canv.sum()

    print("clust1 missplaced:", calcmissmatches(comp1))
    print("clust2 missplaced:", calcmissmatches(comp2))
    print("set1 randindex:",rand(y11,y21) )
    print("set2 randindex:",rand(y12,y22) )







# Clustering  -> use 1NN purity :) 

def clust(nparray, labels):
    neighs = KNC(n_neighbors=2)
    neighs.fit(nparray,labels)
    _, pairs = neighs.kneighbors(nparray)# get neighbor
    acc= sum([labels[a]==labels[b] for a,b in pairs])/len(labels)
    return acc  


def doubleclust(X,X2, Y,Y2):
    def asd(X,X2,Y,Y2):
        neighs = KNC(n_neighbors=1)
        neighs.fit(X,Y)
        _, pairs = neighs.kneighbors(X2)
        return sum([ Y[a]==b  for a,b in zip(pairs,Y2)])/len(Y2)
    
    return (asd(X,X2,Y,Y2)+asd(X2,X,Y2,Y))/2

# preprocessing test
def pptest(distmatrix,matches):
    # return avg dist among hits / meddian dist 
    
    # this should work becausue integer indexing...  
    #return np.mean(distmatrix[*matches])/np.median(distmatrix)
    pass



def mergetest(inst_a, inst_b, labels_a, labels_b, matches):
    # average distance of clusters 
    
    labels = Counter(labsls_a)
    labels.update(Counter(labels_b)) # dunno if this works one might concatenate l_a and l_b somehow
    
    clustercenters_a = [ np.mean(inst_a[labels_a==label], axis=0)  for label in labels]
    clustercenters_b = [ np.mean(inst_b[labels_b==label], axis=0)  for label in labels]
    
    m = pairwise_distances(clustercensters_a, clustercenters_b)
    
    #return np.mean(m[*matches])/np.median(m)
    pass
    
    
    
    
    
    
    
    
