from collections import Counter
from sklearn.model_selection import cross_val_predict
from  sklearn.neighbors import KNeighborsClassifier as KNC
import numpy as np
from sklearn.metrics import pairwise_distances
    
# QUALITY MEASSUREMENTS



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
    
    
    
    
    
    
    
    