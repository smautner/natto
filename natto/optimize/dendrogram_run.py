from natto.input import load
from natto.optimize import distances  as d
import numpy as np 
from functools import partial
from lmz import *
from basics.sgexec import sgeexecuter as sge
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dnames = "human1 human2 human3 human4 fluidigmc1 smartseq2 celseq2 celseq".split()
loaders =  [ partial( load.loadgruen_single, f"../data/punk/{data}",subsample=2000) for data in dnames]


s = sge()
NC = 16

for i in Range(dnames):
    for j in range(i+1, len(dnames)):
        s.add_job( d.rundist_2loaders , [(loaders[i],loaders[j], NC) for r in range(20)])
rr= s.execute()


def dendro(mat, title, fname):
    links = squareform(mat)
    Z = hierarchy.linkage(links, 'single')
    plt.figure()
    hierarchy.dendrogram(Z, labels = dnames)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.savefig(fname, dpi=300)


for z in range(20):
    mtx = np.zeros((len(dnames),len(dnames)))
    rr_index=0
    for i in Range(dnames):
        for j in range(i+1, len(dnames)):
            mtx[i,j]= 1- rr[rr_index][z][0]
            mtx[j,i]= 1- rr[rr_index][z][0]
            rr_index+=1
    dendro(mtx, f"dendrogram run:{z}", f"dendro_{z}.png")






