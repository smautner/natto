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
import x40
import matplotlib
matplotlib.use('module://matplotlib-sixel')
import pandas as pd
import seaborn as sns

def plot(xnames, folder, cleanname):

    filenames = [f"{folder}/varcell{j}.dmp" for j in xnames]
    xdata = Map(tools.loadfile, filenames)
    #xdata2 = Map(lambda x: np.median(x,axis=2), xdata)
    xdata2 = Map(lambda x: x[:,:,4], xdata)
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


def mkDfData(xnames, folder, cleanname):
    filenames = [f"{folder}/500varcell{j}.dmp" for j in xnames]
    xdata = Map(tools.loadfile, filenames)
    # xdata2 = Map(lambda x: np.median(x,axis=2), xdata)
    # xdata2 = Map(lambda x: x[:,:,0], xdata)
    labels,_ = process_labels()
    labels = np.array(labels)
    df_data=[]
    # p@, cleanname, rep, x, y
    for k in [1,2,3]:
        for i in [0,2,3,4,5,6]:# 1 had an error :)
            for xid, x in enumerate(xdata):
                y = precissionK(x[:,:,i],k,labels)
                df_data.append([k,cleanname,i,xnames[xid],y])
    return df_data

def plotsns(data):
    df = pd.DataFrame(data)
    sns.set_theme(style='whitegrid')
    df.columns = 'neighbors±σ method rep genes precision'.split()
    method = df['method'][0]
    title = f'Searching for similar datasets via {method}'
    sns.lineplot(data = df,x='genes', y = 'precision',style= 'neighbors±σ',
            hue = 'neighbors±σ',palette="flare", ci=68)

    plt.title(title, y=1.06, fontsize = 16)
    plt.ylim([.82,1])
    plt.ylabel('precision of neighbors (40 datasets)')
    plt.xlabel('number of cells')
    plt.savefig(f"numcells{method}500genes.png")
    plt.show()
    plt.clf()

if __name__ == "__main__":

    jug = Range(800,4001,400)
    #np.seterr(divide='ignore', invalid='ignore')
    for j in jug:
        file = f'vcell/{j}.dmp'
        if not os.path.exists(file):
            preprocess(7,j,file, njobs = 27)
        x40.preprocess_single_test(7,j,file+"_single")

    for j in jug:
        infile =  f'vcell/{j}.dmp'
        out = f"jacc/500varcell{j}.dmp"
        outcos = f"cosi/500varcell{j}.dmp"
        if not os.path.exists(out):
            calc_mp20(partial(d.jaccard, ngenes=500),out=out, infile=infile, shape = (40,40,7))
            calc_mp20(partial(d.cosine, numgenes=500),out=outcos,infile=infile,shape=(40,40,7))

    plotsns(mkDfData(jug,"cosi","Cosine similarity"))
    plotsns(mkDfData(jug,"jacc","Jaccard similarity"))



