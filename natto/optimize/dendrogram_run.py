from natto.input import load
from natto.optimize import distances  as d
import numpy as np 
from functools import partial
from lmz import *
import basics.sgexec as exe
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def dendro(mat, title, fname):
    links = squareform(mat)
    Z = hierarchy.linkage(links, 'single')
    plt.figure()
    hierarchy.dendrogram(Z, labels = dnames)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.savefig(fname, dpi=300)
    plt.close()
    return Z



dnames = "human1 human2 human3 human4 fluidigmc1 smartseq2 celseq2 celseq".split()
dnames = "human1 human2 human3 human4 smartseq2 celseq2 celseq".split()
loaders =  [ partial( load.loadgruen_single, f"../data/punk/{data}",subsample=1500) for data in dnames]

s = exe.sgeexecuter()
NC = 16
NUMREPS = 3

for i in Range(dnames):
    for j in range(i+1, len(dnames)):
        s.add_job( d.rundist_1nn_2loaders , [(loaders[i],loaders[j], NC) for r in range(NUMREPS)])
rr= s.execute()
#import LOADOLDSHIT
#rr = [exe.collectresults(jid, NUMREPS,True) for jid in LOADOLDSHIT.load()]



def lol(block, s=0): 
    np.array(block).mean(axis=0)[s]


rr_index=0
mtx = np.zeros((len(dnames),len(dnames)))
mtx2 = np.zeros((len(dnames),len(dnames)))
for i in Range(dnames):
    for j in range(i+1, len(dnames)):
        mtx[i,j]=  lol(rr[rr_index])
        mtx[j,i]=  lol(rr[rr_index])
        mtx2[i,j]=  lol(rr[rr_index],1)
        mtx2[j,i]=  lol(rr[rr_index],1)
        rr_index+=1

print (mtx.tolist())
asd = dendro(mtx, f"aasd", f"dendro_baseline.png")
print (mtx2.tolist())
asd = dendro(mtx2, f"aasd asd", f"dendro_baseline2.png")


'''
for z in range(NUMREPS):
    mtx = np.zeros((len(dnames),len(dnames)))
    rr_index=0
    for i in Range(dnames):
        for j in range(i+1, len(dnames)):
            mtx[i,j]= 1- rr[rr_index][z][0]
            mtx[j,i]= 1- rr[rr_index][z][0]
            rr_index+=1
    asd = dendro(mtx, f"dendrogram run:{z}", f"dendro_{z}.png")
    if z == 2:
        print (asd.tolist())
        print (mtx.tolist())


'''




def lol(block): 
    return 1 - np.array(block).mean(axis=0)[0]

'''
rr_index=0
mtx = np.zeros((len(dnames),len(dnames)))
for i in Range(dnames):
    for j in range(i+1, len(dnames)):
        mtx[i,j]=  lol(rr[rr_index])
        mtx[j,i]=  lol(rr[rr_index])
        rr_index+=1
dendro(mtx, f"dendrogram run:SUPER", f"dendro_super_16.png")


rr_index=0
mtx = np.zeros((len(dnames),len(dnames)))
for i in Range(dnames):
    for j in range(i+1, len(dnames)):
        mtx[i,j]=  lol(rr[rr_index])
        mtx[j,i]=  lol(rr[rr_index])
        rr_index+=1
dendro(mtx, f"dendrogram run:SUPER", f"dendro_super_10.png")
'''






