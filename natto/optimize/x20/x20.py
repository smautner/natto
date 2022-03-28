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
import sys
what = sys.argv[1]

from sklearn.metrics import f1_score,adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering

__doc__='''
the point here is to use 5x4 datasets from the 72.
we optimize the parameters, with the objective funktion of increasing the rand score.
we do the grid, and then present the results as table.
'''

def preprocess(types = 5, samples =4, repeats =7, ncells = 1500):
    n = input.get71names()
    datasets = []
    random.seed(43)
    for startswith in 'Testi Colon Prost Tcell Bonem'.split(' ')[:types]:
        data = [d for d in n if d.startswith(startswith)]
        random.shuffle(data)
        datasets+=data[:samples]

    # 1. prep pairwise preprocessed data things...
    loaders =  [ partial( input.load100,
                          data,
                          path = "/home/ubuntu/repos/natto/natto/data",
                          subsample=1500)
                 for data in datasets]

    it = [ [loaders[i],loaders[j]] for i in Range(datasets) for j in range(i+1, len(datasets))]
    def f(loadme):
        a,b = loadme
        print('#'*80)
        print('#'*80)
        print(a,b)
        return [Data().fit([a(),b()],
            visual_ftsel=False,
            pca = 20,
            umaps=[10],
            sortfield = 0,
            make_even=True) for i in range(repeats)]
    return tools.xmap(f,it,32)



if what == 'preprocess':
    #res = preprocess()
    #tools.dumpfile(res,'data.dmp')

    res = preprocess(3,3,3, ncells = 1000) # => 9x9x3 matrix :)
    tools.dumpfile(res,'testdata.dmp')

###############################
###############################
###############################



def calc_mp20(method_label, fname='data.dmp', shape=(20,20,7)):
        # label is the name that we give to the plot.
        # meth is a function that takes a Data object and calculates a similarity
        meth, saveas = method_label
        m = np.zeros(shape,dtype=float)
        data=tools.loadfile(fname)
        data.reverse()
        it = []
        for a in range(m.shape[0]):
            for b in range(m.shape[1]):
                if b > a:
                    for r,dat in enumerate(data.pop()):
                        it.append((a,b,r,dat))
        def f(i):
            b,a,c,data = i
            return b,a,c, meth(data)

        for a,b,c,res in tools.xmap(f,it,32):
            m[b,a,c] = np.median(res)
            m[a,b,c] = m[b,a,c]

        tools.dumpfile(np.median(m,axis=2),f'{saveas}.dmp')


methodlabels = 'Natto PCA#Natto UMAP#Baseline PCA#Baseline Normalized#Baseline UMAP#Base Normalized Hungarian#Baseline PCA Hungarian#Baseline UMAP Hungarian'.split('#')

if what == 'distances':

    baselinemethods = [d.baseline_pca, d.baseline_raw, d.baseline_umap]
    #baselineandhungmethods = [lambda x: d.baseline_hung(x,pid) for pid in [0,1,2]]
    baselineandhungmethods = [partial(d.baseline_hung,pid=p) for p in [0,1,2]]
    #nattomethods = [lambda x: d.natto_distance(x,pid) for pid in [0,1,2]]
    nattomethods = [partial(d.natto_distance,pid =p)  for p in [1,2]]
    methods = nattomethods+baselinemethods+baselineandhungmethods
    # no natto raw as the clusteringis too slow
    methods_labels = Zip(methods, methodlabels)
    #plot_one(methods_labels[0])
    #for ml in methods_labels: plot_one(ml)
    #tools.xmap(calc_one,methods_labels,32)
    for  ml in methods_labels:
        calc_mp20(ml)



from sklearn.cluster import SpectralClustering,KMeans, AgglomerativeClustering


if what == 'distclust':
    '''
    ok so methods... a method takes a data-object and returns a score...
    # the label list should be ez...
    # also i should make a smaller test set.. maybe 9x9x3...
    '''
    methods = []
    methodlabels = []
    meth_to_name = lambda x: type(x(n_clusters=3)).__name__

    for pid in [1,2]:
        #for method in [SpectralClustering, KMeans, AgglomerativeClustering]:
        for method in [ KMeans, AgglomerativeClustering]:
            methods.append(partial(d.clusterAndRari, pid=pid, cluster= method, clusterArgs ={'n_clusters':15}))
            methodlabels.append(meth_to_name(method)+str(pid))

    methods_labels = Zip(methods, methodlabels)
    #tools.xmap(calc_one,methods_labels,32)

    print(methodlabels)
    for  ml in methods_labels:
        calc_mp20(ml,'testdata.dmp',(9,9,3))


###############################3
#
###############################

def score_matrix(X,class_labels,n_clust= 5, method = 'distance'):
        if method == 'similarity' or 'atto' in method:
            X = 1-X
            np.fill_diagonal(X,0)
            print('atto',X)

        pred=  AgglomerativeClustering(n_clusters=n_clust,affinity='precomputed',linkage='complete').fit_predict(X)
        print(f"{method=}{ pred=}")
        return adjusted_rand_score(class_labels,pred)


def plot(methodlabel):
        class_labels = [n for n in range(5) for i in range(4)]
        #print(labels)
        shortnames = 'Testi Colon Prost Tcell Bonem'.split(' ')
        labeldict = {n:text for n,text in zip(range(5),shortnames)}
        m= tools.loadfile(f'{methodlabel}.dmp')
        drawclustermap(m,class_labels,labeldict)

        # pred = AgglomerativeClustering(n_clusters=5,affinity='precomputed',linkage='linkage').fit_predict(m)
        # print(pred, class_labels)
        # print(m)
        # myscores = adjusted_rand_score(class_labels,pred)

        myscores = score_matrix(m,class_labels,method = methodlabel)
        plt.suptitle(methodlabel+f"\n{myscores:.2f}", size = 40, y=1.05, x= .5)
        #bbox_to_anchor=(1,-4.1)
        plt.savefig(f"{methodlabel}.png", bbox_inches='tight')


def plot2(methodlabel):
        class_labels = [n for n in range(3) for i in range(3)]
        #print(labels)
        shortnames = 'Testi Colon Prost'.split(' ')
        labeldict = {n:text for n,text in zip(range(3),shortnames)}

        m= tools.loadfile(f'{methodlabel}.dmp')
        #SQ = squareform(m)
        drawclustermap(m,class_labels,labeldict)

        myscores = score_matrix(m,class_labels,method = methodlabel)
        plt.suptitle(methodlabel+f"\n{myscores:.2f}", size = 40, y=1.05, x= .5)
        #bbox_to_anchor=(1,-4.1)
        plt.savefig(f"{methodlabel}.png", bbox_inches='tight')
        print(f"### {methodlabel} DON3")


if what == 'plot':
    tools.xmap(plot,methodlabels,32)
    # numethods = ['SpectralClustering1', 'KMeans1', 'AgglomerativeClustering1', 'SpectralClustering2', 'KMeans2', 'AgglomerativeClustering2']
    # nu =  [ 'KMeans1', 'AgglomerativeClustering1', 'KMeans2', 'AgglomerativeClustering2']
    #for meth in nu: plot2(meth)
if what == 'plot2':
    nu =  [ 'KMeans1', 'AgglomerativeClustering1', 'KMeans2', 'AgglomerativeClustering2']
    tools.xmap(plot2,nu,32)


