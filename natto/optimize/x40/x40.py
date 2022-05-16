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

def preprocess( repeats =7, ncells = 1500, out = 'data.dmp'):
    datasets = input.get40names()
    random.seed(43)
    loaders =  [ partial( input.load100,
                          data,
                          path = "/home/ubuntu/repos/natto/natto/data",
                          subsample=ncells)
                 for data in datasets]
    it = [ [loaders[i],loaders[j]] for i in Range(datasets) for j in range(i+1, len(datasets))]
    def f(loadme):
        a,b = loadme
        return [Data().fit([a(),b()],
            visual_ftsel=False,
            pca = 0,
            make_readcounts_even=True,
            umaps=[],
            sortfield = -1,
            make_even=True) for i in range(repeats)]
    res =  tools.xmap(f,it,32)
    tools.dumpfile(res,out)

def calc_mp20(meth,out = 'calc.dmp', infile='data.dmp', shape=(40, 40, 5)):
    # label is the name that we give to the plot.
    # meth is a function that takes a Data object and calculates a similarity

    m = np.zeros(shape, dtype=float)
    data = tools.loadfile(infile)
    data.reverse() # because we use pop later
    it = []

    # so we generate the data to be executed [i,j,repeatid,data]
    for a in range(m.shape[0]):
        for b in range(m.shape[1]):
            if b > a:
                for r, dat in enumerate(data.pop()):
                    it.append((a, b, r, dat))

    # execute
    def f(i):
        b, a, c, data = i
        return b, a, c, meth(data)

    for a, b, c, res in tools.xmap(f, it, 32):
        m[b, a, c] = np.median(res)
        m[a, b, c] = m[b, a, c]

    #res = np.median(m, axis=2)
    tools.dumpfile(m, out)


def process_labels():
    shortnames = [n[:5] for n in input.get40names()]
    sm = tools.spacemap(list(set(shortnames)))
    class_labels = [sm.getint[n] for n in shortnames]
    labeldict = sm.getitem
    return class_labels, labeldict

def plot(xnames, folder, cleanname):

    labels = [f"{folder}/{j}.dmp" for j in xnames]
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
        plt.plot(jug, y, label=f'{k} neighbors {cleanname}')



if __name__ == "__main__":
    #preprocess(5,2000)
    jug = Range(50, 1400, 50)
    # for numgenes in jug:
    #     calc_mp20(partial(d.jaccard, ngenes=numgenes),out=f"jacc/{numgenes}.dmp")
    #     calc_mp20(partial(d.cosine, numgenes=numgenes),out=f"cosi/{numgenes}.dmp")
    plot(jug,"cosi","cosine")
    plot(jug,"jacc","jaccard")
    plt.title('Searching for similar datasets')
    plt.ylabel('precision on neighbors 40 datasets')
    plt.xlabel('number of genes')
    plt.legend()
    plt.savefig(f"numgenes.png")


