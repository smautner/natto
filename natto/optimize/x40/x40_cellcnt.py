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
import os
from x40 import preprocess, calc_mp20, process_labels, precissionK
import matplotlib
matplotlib.use('module://matplotlib-sixel')

def plot(xnames, folder, cleanname):

    filenames = [f"{folder}/varcell{j}.dmp" for j in xnames]
    xdata = Map(tools.loadfile, filenames)
    xdata2 = Map(lambda x: np.median(x,axis=2), xdata)
    #xdata2 = Map(lambda x: x[:,:,0], xdata)
    labels,_ = process_labels()
    labels = np.array(labels)

    for k in [1,2,3]: # neighbors
        y = [precissionK(x,k,labels) for x in xdata2]
        print(f"{ y=}")
        plt.plot(jug, y, label=f'{k} cells {cleanname}')

        if k == 0:
            # optionally plitting without averaging, this changes nothing :)
            # just reveals a bug somewhere, probably harmless so we dont track it now
            repeats = xdata[0].shape[2]
            currentslices =[ Map(lambda x: x[:,:,i], xdata) for i in range(repeats)]
            y = [[score(x,k) for x in xdat] for xdat in currentslices]
            plt.scatter(jug*repeats,y, label=f'k={k} {cleanname}')



if __name__ == "__main__":

    jug = Range(400,5001,400)
    #np.seterr(divide='ignore', invalid='ignore')
    for j in jug:
        file = f'vcell/{j}.dmp'
        if not os.path.exists(file):
            preprocess(3,j,file)

    for j in jug:
        infile =  f'vcell/{j}.dmp'
        out = f"jacc/varcell{j}.dmp"
        if not os.path.exists(out):
            calc_mp20(partial(d.jaccard, ngenes=400),out=out, infile=infile)
            calc_mp20(partial(d.cosine, numgenes=400),out=f"cosi/varcell{j}.dmp",infile=infile)

    plot(jug,"cosi","cosine")
    plot(jug,"jacc","jaccard")
    plt.title('Searching for similar datasets')
    plt.ylabel('precision on neighbors 40 datasets')
    plt.xlabel('number of cells sampled')
    plt.legend()
    plt.savefig(f"numcells.png")
    plt.show()

