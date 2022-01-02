from natto.input import load
from natto.optimize import util 
import natto.process as p 
from pprint import pprint
from lmz import *
import numpy as np
from functools import partial
from basics.sgexec import sgeexecuter as sge



"""makes a noise curve"""


def process(level, c):
    l =np.array(level)
    return l.mean(axis = 0 )[c]

def processVar(level, c):
    l =np.array(level)
    return l.var(axis = 0 )[c]

def processstd(level, c):
    l =np.array(level)
    return l.std(axis = 0 )[c]

debug = False
repeats = 40

# select clustering method 
cluster = partial(p.leiden_2,resolution=.5)
cluster = partial(p.gmm_2_dynamic, nc=(4,15))
cluster = partial(p.random_2, nc = 15) 
clusterb = partial(p.spec_2, nc = 15) 
cluster = partial(p.kmeans_2, nc = 15) 
cluster = partial(p.birch_2, nc = 15) 
cluster = partial(p.afprop_2, damping=.5) 
clustera = partial(p.gmm_2, cov='full', nc = 15)


# select DATA
l_3k = partial(load.load3k, subsample=1500)
l_6k = partial(load.load6k, subsample=1500)
l_stim = partial (load.loadpbmc,path='../data/immune_stim/9',subsample=1500,seed=None)
l_p7e = partial(load.loadpbmc, path='../data/p7e',subsample=1500,seed=None)
l_p7d = partial(load.loadpbmc, path='../data/p7d',subsample=1500,seed=None)
l_h1 = partial(load.loadgruen_single, path = '../data/punk/human1',  subsample=1500)
l_h3 = partial(load.loadgruen_single, path = '../data/punk/human3',  subsample=1500)

def run(loader, rname):
    s=sge()
    metrics= ['euclidean']
    metrics= ['euclidean','sqeuclidean','canberra','l1']
    clusternames = ['gmm', 'spec']
    for level in range(0,110,40 if debug else 10):
        s.add_job(util.get_noise_run_moar,
                [(loader, [clustera, clusterb], level, metrics)
                    for r in range(2 if debug else repeats)])

    rr= s.execute()
    #rr=s.load(f"{rname}.sav")

    s.save(f"{rname}.sav")
    
    # ok so we have level x ( [cluster + metrics] x repeats )
    pprint(rr)
    i = 0 
    for i,s in enumerate([f"{c}_{m}" for c in clusternames for m in metrics]):
        mean = [process(level, i) for level in rr]
        std =  [processstd(level, i) for level in rr]
        print(f"a={mean}\nb={std}\n{s}=[a,b,'{s}']")

myloaders=[l_3k, l_6k, l_h1,l_h3,l_p7e, l_p7d]
lnames = ['3k','6k','h1','h3','p7e','p7d']
myloaders=[l_3k]
lnames = ['3k']
for loader,lname in zip(myloaders, lnames):
    run(loader, f'{lname}_g15')
