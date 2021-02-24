#!/home/ubuntu/.myconda/miniconda3/bin/python
import sys
import basics as ba
import numpy as np
from natto.input import load
from natto.input import preprocessing, pp_many
from sklearn.metrics import pairwise_distances, adjusted_rand_score
import ubergauss as ug 
import natto

debug = False
sampnum = 200 if debug else 1000


def get_score(n1, n2, loader, trve, seed):
    sampnum = 200 if debug else 1000
    d1 = loader(n1,seed)
    d2 = loader(n2,seed)
    pp = preprocessing.Data().fit(d1,d2,
                    debug_ftsel=False,
                    scale=True, 
                    do2dumap =False,
                    maxgenes = 800)  
    allteX =  np.vstack(pp.dx)
    print("cluster start")
    labels = natto.process.gmm_1(allteX,nc=15)
    print("cluster done")
    #labels = ug.get_model(allteX, poolsize=1,nclust_min=7, nclust_max=17).predict(allteX)
    #labels = ug.get_model(allteX, poolsize=1,nclust_min=12, nclust_max=12).predict(allteX)
    real_labels = trve(pp)
    #print (real_labels, labels)
    rands = adjusted_rand_score(real_labels, labels)
    return rands


def get_params_100():
    trve = lambda pp:  [i for d in [pp.a, pp.b] for i in d.obs['true'].values]
    dnames = """Testis_10xchromium_SRA645804-SRS2823404_4197.h5  Testis_10xchromium_SRA645804-SRS2823409_4791.h5
    Testis_10xchromium_SRA645804-SRS2823405_3598.h5  Testis_10xchromium_SRA645804-SRS2823410_4045.h5
    Testis_10xchromium_SRA645804-SRS2823406_3989.h5  Testis_10xchromium_SRA645804-SRS2823412_5299.h5
    Testis_10xchromium_SRA645804-SRS2823407_4046.h5  Testis_10xchromium_SRA645804-SRS3572594_4574.h5
    Testis_10xchromium_SRA645804-SRS2823408_4306.h5"""
    dnames = [d[:-3] for d in dnames.split()]
    loader = lambda x, seed : load.load100(x,subsample=sampnum,path='../data', remove_unlabeled = True, seed = seed)
    return dnames, loader, trve

def get_params_punk():
    trve = lambda pp:  [i for d in [pp.a, pp.b] for i in d.obs['true'].values]
    dnames = "human1 human2 human3 human4 smartseq2 celseq2 celseq".split()
    loader = lambda x, seed: load.loadgruen_single(f"../data/punk/{x}", subsample=sampnum, seed = seed)
    return dnames, loader, trve



if __name__ == "__main__":
    task,t2,rep = map(int, sys.argv[1].strip().split(' '))
    dnames, loader, trve  = get_params_100()
    other = dnames[t2]
    self = dnames[task]
    result=  get_score(self, other,loader,trve, seed = rep)
    print("res: ", result)
    ba.dumpfile(result,"res/"+sys.argv[1].replace(" ",'_'))
    print("all good")


# use median instead of mean! TODO 
# add error baro to plot -> quantiles! plot points! fit forrcoeff linear 
# check the seed value for subsampling
# look at 100
def res(indices,reps): 
    print (dnames)
    for i in range(indices):
        indexrepeats =  np.array([ba.loadfile(f"res/{i}_{r}") for r in range(reps) ]) 
        print ( indexrepeats.mean(axis=0).tolist())
