from lmz import Map,Zip,Filter,Grouper,Range,Transpose
import structout as so
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
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.cluster import SpectralClustering,KMeans, AgglomerativeClustering


__doc__='''

ok the plan is to load the timeseries data and draw a 4x4 heatmap

'''


def calc(d1,d2,meth):
    d1 = d1.subsample(inplace=False)
    d2 = d2.subsample(inplace=False)
    data = Data().fit([d1,d2], visual_ftsel = False, pca= 20, umaps=[10])
    return meth(data)

if what == 'fag':
    alldatasets = input.
    l= len(alldatasets)
    distances  = np.ones((l,l))
    for i in range(l):
        for j in range(i+1,j):
            r = calc(alldatasets[i], alldatasets[j], meth= lambda x: d.cosine(x,600))
            distances[i,j]= r
            distances[j,i]= r
    so.heatmap(distances)




def preprocess( repeats =7, ncells = 1500, numgenes= 800):
    datasets = input.get57names()
    random.seed(43)


    loaders =  [ partial( input.load100,
                          data,
                          path = "/home/ubuntu/repos/natto/natto/data",
                          subsample=ncells)
                 for data in datasets]

    it = [ [loaders[i],loaders[j]] for i in Range(datasets) for j in range(i+1, len(datasets))]
    def f(loadme):
        a,b = loadme
        print('#'*80)
        print('#'*80)
        print(a,b)
        return [Data().fit([a(),b()],
            visual_ftsel=False,selectgenes=numgenes,
            pca = 20,
            make_readcounts_even=True,
            umaps=[10],
            sortfield = -1,
            make_even=True) for i in range(repeats)]
    return tools.xmap(f,it,32)



if what == 'preprocess':
    #res = preprocess()
    #tools.dumpfile(res,'data.dmp')
    jug = Range(200,1200,200)
    for ng in jug:
        res = preprocess(repeats=5, ncells = 2000,numgenes=ng)
        tools.dumpfile(res,'natt/ng{ng}.dmp')

###############################
###############################



def calc_mp20(method_label, fname='data_pcaUmap.dmp', shape=(57,57,5)):
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

        res = np.median(m,axis=2)
        tools.dumpfile(res,f'{saveas}.ddmp')
        return res


methodlabels = 'natto#cosine DEGenes#jaccard DEGenes#HUNG'.split('#')
if what == 'distances':
    methods_labels = Zip([ partial(d.natto_distance, pid=2),d.cosine,d.jaccard,partial(d.baseline_hung, pid=0)], methodlabels)
    for  ml in methods_labels:
        #calc_mp20(ml,shape=(5,5,5))
        r=calc_mp20(ml,fname='data_pcaUmap.dmp')
        break




def calc_sp(method_label, fname='data_pcaUmap.dmp', shape=(57,57,5)):
        # label is the name that we give to the plot.
        # meth is a function that takes a Data object and calculates a similarity
        meth, saveas = method_label
        m = np.zeros(shape,dtype=float)
        data=tools.loadfile(fname)
        data.reverse()
        it = []
        res = np.zeros(shape[:2],dtype=np.float64)
        for a in range(m.shape[0]):
            for b in range(m.shape[1]):
                if b > a:
                    dat = data.pop()
                    z = np.median(Map(meth,dat))
                    res[b,a] = z
                    res[a,b] = z
        tools.dumpfile(res,f'{saveas}.ddmp')
        print(saveas)
        return res


if what == 'jaccard':
    jug = Range(50,1400,50)
    labels = [f"jacc/{j} genes" for j in jug]
    methods = [partial(d.jaccard,ngenes=j) for j in jug]
    #tools.xmap(calc_sp, zip(methods, labels),processes=5)
    for  ml in zip(methods,labels):
        #calc_mp20(ml,shape=(5,5,5))
        r=calc_mp20(ml)

if what == 'cosine':
    r=calc_mp20((d.cosine,'cosi/cosine'))

if what == 'cosine2':
    jug = Range(50,1400,50)
    labels = [f"cosi/{j} genes" for j in jug]
    methods = [partial(d.cosine,numgenes=j) for j in jug]
    #tools.xmap(calc_sp, zip(methods, labels),processes=5)
    for  ml in zip(methods,labels):
        #calc_mp20(ml,shape=(5,5,5))
        r=calc_mp20(ml)

if what == 'natto2':
    jug = Range(200,1200,200)
    for ng in jug:
        method = partial(d.natto_distance,pid=-1)
        label = f"natt/{ng}gen"
        ml =  method, label
        r=calc_mp20(ml, fname = 'natt/ng{ng}.dmp')




if what == 'makeblob':
    alldistances=[]
    alldistances = [tools.loadfile()]
    tools.dumpfile(alldistaces,'uneven.dmp')





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


def score_matrix_rand(X,class_labels,n_clust= 5):
        pred=  AgglomerativeClustering(n_clusters=n_clust,
                affinity='precomputed',linkage='complete').fit_predict(X)
        score = adjusted_rand_score(class_labels,pred)
        print(pred)
        print(class_labels)
        print(score)
        return score

def score_matrix_f1_affinity(X,class_labels,n_neigh=3):
    class_labels = np.array(class_labels)
    true = np.hstack([class_labels for i in range(n_neigh)])
    np.fill_diagonal(X,np.NINF)
    srt = np.argsort(X,axis=1)
    pred = np.hstack([np.take(class_labels,srt[:,-(n+1)])for n in range(n_neigh)])

    if False:
        from scipy.stats import mode
        pred = mode(class_labels[srt][:,:n_neigh], axis = 1)
        pred = pred[0].flatten()


    #score =  f1_score(true,pred,average='micro')
    from sklearn.metrics import precision_score
    score =  precision_score(true,pred,average='micro')


    #print(true,pred,score)
    # so.lprint(true)
    # so.lprint(pred)
    return score



def process_labels():
        shortnames = [ n[:5] for n in input.get57names()]
        sm = tools.spacemap(list(set(shortnames)))
        class_labels = [sm.getint[n] for n in shortnames]
        labeldict = sm.getitem
        return class_labels, labeldict

def plot(methodlabel):
        #print(labels)

        class_labels, labeldict = process_labels()
        m= tools.loadfile(f'{methodlabel}.ddmp')

        ## B
        if  "HUNG" in methodlabel:
            m = 1-m
        np.fill_diagonal(m,1)

        clustergrid = drawclustermap(m,class_labels,labeldict)
        mat = clustergrid.data2d.to_numpy()
        #so.heatmap(mat)

        # pred = AgglomerativeClustering(n_clusters=5,affinity='precomputed',linkage='linkage').fit_predict(m)
        # print(pred, class_labels)
        # print(m)
        # myscores = adjusted_rand_score(class_labels,pred)

        #myscores = score_matrix(m,class_labels,n_clust = sm.len)
        myscores = score_matrix_f1_affinity(np.array(m),class_labels,n_neigh=3)
        plt.suptitle(methodlabel+f"\n{myscores:.2f}", size = 40, y=1.05, x= .5)
        #bbox_to_anchor=(1,-4.1)
        plt.savefig(f"{methodlabel}.png", bbox_inches='tight')
        so.heatmap(m)
        return myscores



if what == 'plot':
    #tools.xmap(plot,methodlabels,32)
    odd = ['uneven/'+m for m in methodlabels]
    even = ['even/'+m for m in methodlabels]
    res=[]
    for ml in odd+even:
        try:
            score = plot(ml)
            res+=[(ml,score)]
        except:
            print(ml)

    for meth,score in res:
        print (meth,'\t',score)

if what == 'evaljacc':
    jug = Range(50,1400,50)
    labels = [f"jacc/{j} genes" for j in jug]
    class_labels, _ = process_labels()


    more_labels = [f"cosi/{j} genes" for j in jug]

    ###
    # pl
    ####
    def plot_numgene(labels):
        allscores = []
        neighborvalues = [1,2,3]
        for neigh in neighborvalues:
            scores = []
            for f in labels:
                m= tools.loadfile(f'{f}.ddmp')
                if neigh == 3:
                    so.heatmap(m)
                scores.append( score_matrix_f1_affinity(np.array(m),class_labels,n_neigh=neigh))
            allscores.append(scores)
        allscores = np.array(allscores)
        for i,row in enumerate(allscores):
            plt.plot(jug,row, label = f'{i+1} neighbors {labels[0][:4]}')

    plot_numgene(labels)
    plot_numgene(more_labels)


    if False: # add cosine sim
            m= tools.loadfile(f'cosi/cosine.ddmp')
            scoress = [ score_matrix_f1_affinity(np.array(m),class_labels,n_neigh=neigh) for neigh in [1,2,3]]
            for i,e in enumerate(scoress):
                plt.scatter([0],[e], label = f'{i+1} neighbors -- cosine')

    if True: # add cosine sim, but mix in percentage wise
            mcos= tools.loadfile(f'cosi/cosine.ddmp')
            mjac= tools.loadfile(f'jacc/600 genes.ddmp')
            mcos = mcos*(mjac.sum()/mcos.sum())
            jug = [0]+jug
            scoress = [ [score_matrix_f1_affinity(mjac*j + mcos*(1-j),class_labels,n_neigh=neigh) for j in [j/max(jug) for j in jug] ] for neigh in [1,2,3]]
            for i,e in enumerate(scoress):
                plt.plot(jug,e, linestyle = ':',label = f'{i+1} neighbors -- 600 cosine decreases')



    plt.title('jaccard DE gene overlap')
    plt.ylabel('score f1-x neighbors 57datasets')
    plt.xlabel('number of genes')
    plt.legend()
    plt.savefig("jacc2419.png")


if what == 'fck':
    class_labels, _ = process_labels()
    from collections import Counter
    print(Counter(class_labels))
