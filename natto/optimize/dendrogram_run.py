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
#dnames = "human1 smartseq2 celseq2 celseq".split()
loaders =  [ partial( load.loadgruen_single, f"../data/punk/{data}",subsample=600) for data in dnames]
s = exe.sgeexecuter()

NUMREPS = 10
SCALEPARAMS = [True]# [True, False]
PCAPARAMS = [20] #[50,40,30,20]
for scale in SCALEPARAMS:
    for pca  in PCAPARAMS:
        for i in Range(dnames):
            for j in range(i+1, len(dnames)):
                #s.add_job( d.rundist_1nn_2loaders , [(loaders[i],loaders[j], pca,scale) for r in range(NUMREPS)])
                s.add_job( d.normal, [(loaders[i],loaders[j], pca, scale) for r in range(NUMREPS)])


rr= s.execute()
s.save("dendro_1nn")
#s = exe.sgeexecuter(load='dendro_1nn')
#rr= s.collect()

print(rr)

#import LOADOLDSHIT
#rr = [exe.collectresults(jid, NUMREPS,True) for jid in LOADOLDSHIT.load()]

def lol(block, s=0): 
    return np.array(block).mean(axis=0)[s]


# print average
rr_index=0
mtx = np.zeros((len(dnames),len(dnames)))
for i in Range(dnames):
    for j in range(i+1, len(dnames)):
        avg_value =  lol(rr[rr_index]) 
        mtx[i,j]= avg_value
        mtx[j,i]=  avg_value
        #print(i,j, np.array(rr[rr_index]).mean(axis=0)[0], avg_value)
        rr_index+=1
print("mtx_all")
print (mtx.tolist())

asd = dendro(mtx, f"aasd", f"dendro_baseline.png")


# print some dendros
for r in range(NUMREPS):
    rr_index=0
    mtx = np.zeros((len(dnames),len(dnames)))
    for i in Range(dnames):
        for j in range(i+1, len(dnames)):
            mtx[i,j]=  rr[rr_index][r][0]
            mtx[j,i]=  rr[rr_index][r][0]
            rr_index+=1
    print(f"mtx_{r}")
    print (mtx.tolist())
    asd = dendro(mtx, f"aasd", f"dendro_baseline{r}.png")



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



'''

def lol(block, i =0): 
    return 1 - np.array(block).mean(axis=0)[i]

def lolstd(block, i =0): 
    return np.array(block).std(axis=0)[i]

rr_index=0
mtx = np.zeros((len(dnames),len(dnames)))
for i in Range(dnames):
    for j in range(i+1, len(dnames)):
        mtx[i,j]=  lol(rr[rr_index])
        mtx[j,i]=  lol(rr[rr_index])
        rr_index+=1
dendro(mtx, f"dendrogram run:SUPER", f"dendro_super_16.png")


def mkdendro(rindex, dendroname, schpalde=0):
    mtx = np.zeros((len(dnames),len(dnames)))
    for i in Range(dnames):
        for j in range(i+1, len(dnames)):
            mtx[i,j]=  lol(rr[rindex], i=schpalde)
            mtx[j,i]=  lol(rr[rindex], i=schpalde)
            d=mtx[i,j]
            s = lolstd(rr[rindex])
            print (f"{dnames[i]}-{dnames[j]} : {d}+-{s}")
            rindex+=1
    dendro(mtx, f"{dendroname}", f"{dendroname}.png")
    print(mtx.tolist())
    return rindex



rr_index= 0
for scale in SCALEPARAMS:
    for pca  in PCAPARAMS:
        rr_index = mkdendro(rr_index, f"S_{scale}_pca_{pca}_map2_20")



'''
'''
for ix, nc in enumerate(NC): 
    rr_index=0
    mtx = np.zeros((len(dnames),len(dnames)))
    for i in Range(dnames):
        for j in range(i+1, len(dnames)):
            mtx[i,j]=  lol(rr[rr_index], i=ix)
            mtx[j,i]=  lol(rr[rr_index], i=ix)
            d=mtx[i,j]
            s = lolstd(rr[rr_index])
            print (f"{dnames[i]}-{dnames[j]} : {d}+-{s}")
            rr_index+=1
    #dendro(mtx, f"dendrogram run:SUPER{nc}", f"dendro_super_multinc_{nc}.png")
    print(mtx.tolist())
'''







