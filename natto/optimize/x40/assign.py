from lmz import Map,Zip,Filter,Grouper,Range,Transpose
import x40_dmp as dmp
from natto import input, process
from sklearn.metrics import adjusted_rand_score
import natto.out.draw as draw
import matplotlib
matplotlib.use('module://matplotlib-sixel')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from scipy.optimize import linear_sum_assignment as lsa
from sklearn import neighbors as nbrs, metrics
from scipy.sparse.csgraph import dijkstra
from ubergauss import tools


'''
1. load the dict a => b,c OK
2. load the data           OK
3. joint pp (either b or b,c) OK

4. diffuse the real labels from b,, assign labels blabla

5. eval

'''


def getzedata(li,neighs=1,numcells=1500, seed = 31337):
    datasetnames = li[:neighs+1]
    zedata= [input.load100(x,
                          path = "/home/ubuntu/repos/natto/natto/data",
                          seed = seed,
                          remove_unlabeled= True,
                          subsample=numcells) for x in datasetnames]

    zedata =  process.Data().fit(zedata,
            visual_ftsel=False,
            pca = 20,
            make_readcounts_even=True,
            umaps=[10,2],
            sortfield = 0,# real labels need to follow the sorting i think...
            make_even=True)

    truelabels = [np.array(zedata.data[x].obs['true']) for x in [0,1]]
    return datasetnames, zedata, truelabels

def confuse(y1,y2, norm = True):
    #cm = confusion_matrix(y1,y2)
    cm = confusion_matrix(y1,y2)
    _, new_order = lsa(-cm)
    cm = cm[new_order]

    xlab,xcounts = np.unique(y2,return_counts = True)
    ylab,ycounts = np.unique(y1,return_counts=True)
    ylab,ycounts = ylab[new_order], ycounts[new_order]

    if norm:
        cm = cm.astype(float)
        for x in Range(xlab):
            for y in Range(ylab):
                res = (2*cm[y,x]) / (xcounts[x]+ycounts[y])
                #print(cm[y,x])
                #print(xcounts[x],ycounts[y])
                print(res)
                cm[y,x]  = res
    print(cm)
    sns.heatmap(cm,xticklabels=xlab,yticklabels=ylab, annot=False,
            linewidths=.5,cmap="YlGnBu" , square=True)
    plt.xlabel('Clusters data set 2')
    plt.ylabel('Clusters data set 1')

def confuse2(y1,y2):
    f=plt.figure(figsize=(16,8))
    ax=plt.subplot(121,title ='absolute hits')
    confuse(y1,y2,norm=False)
    ax=plt.subplot(122,title ='relative hits (2x hits / sumOfLabels)')
    confuse(y1,y2,norm=True)
    plt.show()


neighbors = dmp.neighs(draw=False)

def hungmat(x1,x2):
        x= metrics.euclidean_distances(x1,x2)
        r = np.zeros_like(x)
        a,b = lsa(x)
        r[a,b] = 1

        r2 = np.zeros((x.shape[1], x.shape[0]))
        r2[b,a] = 1 # rorated :)
        return r,r2



def mykernel(x1len=False,neighbors = 3, X=None,_=None, return_graph = False):
    assert x1len, 'need to know how large the first dataset ist :)'
    '''
    X are the stacked projections[0] (normalized read matrices)
    since this is a kernel, we return a similarity matrix

    - we can split it by 2 to get the original projections
    - we do neighbors to get quadrant 2 and 4
    - we do hungarian to do quadrants 1 and 3
    - we do dijkstra to get a complete distance matrix

    '''
    x1,x2 = np.split(X,[x1len])
    print(f"{ x1.shape=}")
    print(f"{ x2.shape=}")
    q2 = nbrs.kneighbors_graph(x1,neighbors).todense()
    #q2 = nbrs.NearestNeighbors(n_neighbors=neighbors, algorithm='brute').fit(x1).kneighbors_graph().todense()
    q4 = nbrs.kneighbors_graph(x2,neighbors).todense()
    q1,q3 = hungmat(x1,x2)

    graph = np.hstack((np.vstack((q2,q3)),np.vstack((q1,q4))))


    connect = dijkstra(graph,unweighted = True, directed = False)

    if return_graph:
        return graph, connect
    distances = -connect # invert
    distances -= distances.min() # longest = 4
    distances /= distances.max() # between 0 and 1 :)

    return distances

def testkernel():
    x1len = 4
    neighbors = 1
    X = np.array([[x,x] for x in Range(x1len)+Range(2)])
    g,d = mykernel(x1len=x1len, neighbors=neighbors, X=X, _=X, return_graph=True)
    print(g)
    print(d)


def diffuse(data, y1, neighbors = 7):
    from sklearn.semi_supervised import LabelSpreading
    lp_model = LabelSpreading(kernel = lambda x,y: mykernel(data.projections[0][0].shape[0],neighbors,x,y) )
    args = np.vstack(Map(tools.zehidense,data.projections[2])), y1
    lp_model.fit(*args)
    return  lp_model.transduction_



if __name__=='__main__':
    for i in [10]:

        names, zedata, truelabels = getzedata(neighbors[i], neighs = 1, numcells = 1000)
        draw.cmp2(*truelabels,*zedata.d2)
        plt.close()
        # print(f"annotation score: {adjusted_rand_score(*truelabels)}")
        # print(f"{names=}")

        # y1,y2 = [np.array([-1,  2,  6,  1, -1, -1, -1, -1, -1, -1,  3,  2,  2, -1, -1,  4,  4, -1, -1, -1,  5, -1, -1, -1, -1,  0,  2,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1,  2, -1,  6, -1, -1,  1, -1,  0, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  5, -1,  0,  4,  9, -1, -1, -1, -1, 2,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  5, -1, -1, -1, 8, -1, -1, -1, -1, -1,  5, -1,  5,  0, -1, -1, -1, -1, -1, -1, -1, -1,  2,  0, -1,  0,  4, -1, -1, -1,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0, -1, -1, -1,  1, -1,  0,  6, -1, -1,  7, -1, -1,  7, -1,  7, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1,  2, -1,  1, -1, -1, -1, -1, -1,  4,  7, -1, -1, -1,  8, -1,  1,  0,  4, -1,  0, -1, -1, -1,  4, -1, -1,  6, -1, -1, -1, -1, -1, -1, -1,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  5, -1, -1,  7, -1, -1, -1, 0, -1,  7, -1,  0, -1, -1, -1,  5, -1, -1, -1, -1, -1, -1, -1,  2, 6, -1, -1, -1, -1, -1, -1,  1, -1,  6, -1, -1, -1, -1, -1,  4,  1, -1,  4, -1, -1, -1,  2, -1, -1, -1, -1, -1, -1, -1,  3,  1,  1, -1, -1, -1, -1,  2, -1,  3, -1, -1,  3, -1,  0, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0, -1,  9, -1, -1, -1, 10, -1, -1, -1,  5, -1, -1,  4,  2, -1, -1, -1, -1, -1, -1,  1,  1, -1, 3, -1, -1, -1, -1,  2, -1, -1,  3, -1,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  5, -1, -1,  2,  2, -1,  1, -1, -1, -1, -1,  0, -1, -1, -1, -1,  0, -1, -1, -1, -1, -1, 10,  6, -1, -1, -1, -1, -1,  0, -1, -1, -1,  1, -1,  2,  1, -1,  7, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1,  7,  7, -1, -1,  2, -1, -1, -1, -1, -1,  6, -1, -1,  8, 10, -1, -1, -1, -1,  5, -1, -1, -1, -1, -1, -1, -1, -1,  6,  1, -1, 8,  2, -1,  0,  4, -1,  0, -1, -1, -1, -1, -1,  2,  2, -1, -1,  1, -1, -1, -1, -1, -1, -1,  1,  2, -1,  1, -1,  9, -1,  8,  4,  4, -1, -1,  1, -1, -1, -1, -1, -1,  0, -1, -1, -1, -1,  4, -1, -1, -1, -1, -1,  4, -1,  4, -1, -1,  8, -1,  2,  0,  0, -1, -1, -1, -1,  0, -1, 0, -1, -1, -1,  0, -1,  0,  4,  1, -1,  2, -1, -1,  3,  2, -1,  0, 9, -1,  1, -1, -1, -1, -1, -1,  0, -1, -1, -1,  5, -1,  0, -1,  1, -1, -1, -1, -1,  2, -1, -1, -1,  6, -1, -1,  1, -1, -1, -1,  5, -1, -1,  1, -1,  7,  3, -1, -1, -1, -1, -1,  3, -1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  7, -1,  1, -1, -1, -1,  1, 1, -1, -1,  0,  3, -1,  1, -1, -1, -1,  0,  4,  3, -1,  6,  1, -1, 8,  6,  2,  9, -1,  5,  3, -1,  5, -1, -1, -1,  1,  1, -1,  8, -1, -1, -1, -1, -1, -1,  2, -1,  4, -1, -1, -1, -1,  2, -1, -1,  5, -1, 1,  1, -1, -1,  2, -1, -1, -1, -1,  8, -1, -1, -1,  9,  0,  4,  7, -1, -1, -1, -1,  8, -1,  0,  5,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  5,  4,  0, -1, -1,  2, 0,  0, -1,  8, -1, -1, -1, -1, -1,  0,  9, -1, -1, -1,  2,  6,  7, -1,  5, -1,  0]), np.array([-1,  1,  5,  3, -1, -1, -1, -1, -1, -1, -1,  1,  7, -1, -1,  0,  0, -1,  1,  9, -1, -1, -1,  4, -1,  0, -1, -1, -1, -1, -1,  4, -1, -1, 8, -1,  0,  1, -1, -1, -1, -1,  2, -1,  0, -1,  2, -1, -1, -1, -1, -1,  0, -1, -1, -1, -1, -1, -1, -1,  0,  3,  0,  0, -1, -1, -1, -1, 1,  8, -1, -1, -1,  4,  6, -1, -1, -1, -1, -1, -1,  6,  3, -1, -1, 2,  1, -1, -1, -1, -1, -1, -1,  9,  0,  0,  8, -1,  8, -1, -1, -1, -1, -1,  3, -1,  3,  0,  0, -1, -1,  1,  6, -1, -1, -1, -1, -1, -1, 1, -1, -1,  0, -1, -1, -1, -1, -1,  0,  1, -1, -1,  1, -1,  2,  3, -1,  1, -1,  5, -1, -1,  3,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,  4,  7, -1, -1, -1,  2, -1, -1,  4,  3, -1,  0, -1, -1, -1,  4, -1, -1, -1,  1, -1, -1, -1, -1,  0, -1,  3, -1, -1,  0,  3, -1, -1, -1, -1,  4, -1,  9, -1, -1, -1, -1, -1, -1,  4, -1, -1, -1, 3, -1,  7, -1,  4, -1,  0, -1,  9, -1,  2, -1,  2,  3, -1, -1,  1, 7, -1, -1, -1, -1, -1, -1,  2, -1,  7,  2, -1, -1, -1, -1,  3,  3, -1,  3, -1, -1,  0, -1, -1,  4, -1, -1, -1, -1, -1, -1,  1, -1, -1, 0, -1, -1,  1, -1, -1,  0, -1, -1,  5,  0,  2, -1,  3, -1, -1, -1, 8, -1, -1, -1, -1,  4, -1, -1, -1, -1,  3, -1,  0, -1, -1, -1,  7, -1, -1, -1,  6, -1, -1,  0, -1, -1, -1,  6, -1, -1, -1,  0,  3, -1, 0, -1, -1, -1,  6, -1, -1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, 8, -1, -1, -1, -1, -1,  1, -1, -1, -1,  1,  1,  0,  2, -1, -1, -1, 4,  3, -1, -1, -1, -1,  0,  1, -1, -1, -1, -1, -1,  5, -1, -1, -1, -1,  4, -1, -1, -1, -1, -1, -1,  1, -1, -1,  7, -1,  6, -1, -1, -1, -1, -1, -1,  5,  1, -1, -1, -1, -1,  2, -1,  5,  4, -1,  1, -1, -1, -1, -1, -1,  5,  4, -1, -1,  1, -1, -1, -1, -1,  2,  5,  1, -1,  2, 7, -1, -1, -1, -1,  6, -1,  5, -1, -1, -1,  1, -1, -1,  0,  3, -1, -1, -1, -1,  3,  0,  2,  0, -1, -1, -1,  0, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  2, -1, -1,  6,  2,  0,  4, -1, -1,  2,  1, -1, -1, -1,  7,  4, -1, -1, -1, -1,  3, -1,  4, -1, -1, 5,  4,  1,  4,  1, -1, -1, -1,  1,  0,  0, -1, -1, -1, -1,  0, -1, -1, -1, -1,  0, -1,  0,  0,  0,  2, -1, -1, -1,  5,  8,  1,  2,  4, 0, -1,  2, -1, -1, -1, -1,  0,  0, -1, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0, -1,  4,  1, -1,  2,  7, -1, -1, -1, -1, 0,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  8, -1, -1, -1,  2, -1, -1, -1,  2, -1, -1, -1,  8, -1, -1, -1, -1, -1,  5, 2,  0, -1, -1,  8, -1, -1,  2, -1, -1,  0, -1,  0, -1,  5, -1, -1, 2,  0, -1,  3, -1, -1,  8, -1, -1,  0,  2, -1,  2,  2, -1,  3, -1, 5, -1, -1, -1, -1, -1, -1,  3, -1,  4,  4, -1,  1, -1, -1, -1, -1, 3,  2, -1, -1,  5, -1,  2, -1, -1,  2, -1, -1, -1,  3, -1,  0,  3, -1, -1, -1, -1,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0, -1,  8, -1, -1,  6, -1, -1, -1, -1, -1, -1, -1,  4, -1, -1, -1, -1, 0,  0, -1,  2,  4, -1,  2, -1, -1, -1,  3, -1, -1, -1,  3,  5,  7, -1,  9, -1, -1])]
        # mask = np.logical_and(y1!=-1, y2!=-1)
        # y1,y2 = y1[mask], y2[mask]
        # confuse2(y1,y2)
        # plt.tight_layout()
        # plt.show()

        # then we diffuse
        nula = diffuse(zedata, np.hstack((truelabels[0], np.full_like(truelabels[0],-1))))
        l1,l2  = np.split(nula,2)
        draw.cmp2(l1,l2,*zedata.d2)

        # then we do it again and compare the rand scores








