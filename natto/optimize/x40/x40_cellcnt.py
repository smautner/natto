from lmz import Map,Zip,Filter,Grouper,Range,Transpose
from natto.optimize import util  as d
from natto import input
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from natto.process import Data
from natto.optimize.plot.dendro import drawclustermap
from natto import process
from ubergauss import tools
import random
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.cluster import SpectralClustering,KMeans, AgglomerativeClustering
from sklearn.metrics import precision_score
import structout as so
from x40 import preprocess, calc_mp20, process_labels


def plot(xnames, folder, cleanname):

    labels = [f"{folder}/varcell{j}.dmp" for j in xnames]
    xdata = map(tools.loadfile, labels)
    # xdata = Map(lambda x: np.median(x,axis=2), xdata)
    xdata = Map(lambda x: x[:,:,0], xdata)
    labels,_ = process_labels()
    labels = np.array(labels)

    def score(m,k):
        true = np.hstack([labels for i in range(k)])
        srt = np.argsort(m, axis=1)
        #pred = labels[ [ srt[i,-j] for i in Range(labels) for j in range(k)] ]
        pred = labels[ [ srt[i,-j]  for j in range(k) for i in Range(labels)] ]
        return  precision_score(true, pred, average='micro')

    for k in [1,2,3]: # neighbors
        y = [score(x,k) for x in xdata]
        plt.plot(jug, y, label=f'{k} cells {cleanname}')



if __name__ == "__main__":

    jug = Range(200,3001,400)
    np.seterr(divide='ignore', invalid='ignore')
    for j in jug:
        preprocess(3,j, f'vcell/{j}.dmp')

    for j in jug:
        infile =  f'vcell/{j}.dmp'
        calc_mp20(partial(d.jaccard, ngenes=400),out=f"jacc/varcell{j}.dmp", infile=infile)
        calc_mp20(partial(d.cosine, numgenes=400),out=f"cosi/varcell{j}.dmp",infile=infile)

    plot(jug,"cosi","cosine")
    plot(jug,"jacc","jaccard")
    plt.title('Searching for similar datasets')
    plt.ylabel('precision on neighbors 40 datasets')
    plt.xlabel('number of cells sampled')
    plt.legend()
    plt.savefig(f"numcells.png")


