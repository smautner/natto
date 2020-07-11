
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def clusters(data,minclust = 3, maxpoison=.125,debug=False):
    
    while minclust < 25:
        r, poison = cluster(data, minclust, debug)
        if poison < maxpoison:
            okcluster = r
        else:
            return okcluster
        minclust+=1
    print("tried maxcluster, stopping")
    return okcluster

def cluster(data, numc,debug):
 
    model = SpectralClustering(n_clusters = numc)
    res= model.fit_predict(data)
    poison = poisonstats(data,res, debug)
    return res, max(poison)
    
    
def poisonstats(data,y, debug=False):
    model =KNeighborsClassifier(n_neighbors=6)
    model.fit(data,y)
    neighs = model.kneighbors( data, n_neighbors=6, return_distance=False)
    res=[]
    for cls, cnt  in  zip(*np.unique(y, return_counts=True)):
        idx = y==cls
        myneighs=neighs[idx,:cnt]
        myneighs = myneighs.flatten()
        neighcls = (y[myneighs])
        problems= sum(neighcls!=cls)
        if debug:
            print(f"cls:{cls}, cnt{cnt}, problems:{problems}")
        res.append(problems/cnt)
    return res
