import basics as ba 
from natto.input import load 
from natto.optimize import noise  as n 
import numpy as np 
from functools import partial
import natto.process as p 
from basics.sgexec import sgeexecuter as sge

def process(level, c):
    l =np.array(level)
    return l.mean(axis = 0 )[c]

def processVar(level, c):
    l =np.array(level)
    return l.var(axis = 0 )[c]

def processstd(level, c):
    l =np.array(level)
    return l.std(axis = 0 )[c]

cluster = partial(p.leiden_2,resolution=.5)
cluster = partial(p.gmm_2, cov='full', nc = 8)
cluster = partial(p.gmm_2_dynamic, nc=(4,15))
cluster = partial(p.random_2, nc = 15) 
cluster = partial(p.spec_2, nc = 15) 
cluster = partial(p.kmeans_2, nc = 15) 
cluster = partial(p.birch_2, nc = 15) 
cluster = partial(p.afprop_2, damping=.5) 


l_3k = partial(load.load3k, subsample=1500)
l_6k = partial(load.load6k, subsample=1500)
#l_stim = partial (load.loadpbmc,path='../data/immune_stim/9',subsample=1500,seed=None)
l_p7e = partial(load.loadpbmc, path='../data/p7e',subsample=1500,seed=None)
l_p7d = partial(load.loadpbmc, path='../data/p7d',subsample=1500,seed=None)
l_h1 = partial(load.loadgruen_single, path = '../data/punk/human1',  subsample=1500)
l_h3 = partial(load.loadgruen_single, path = '../data/punk/human3',  subsample=1500)

NUMREPEATS = 2
def run(loader, rname):
    s=sge(loglevel= 70) 
    for level in range(0,110,10):
        s.add_job( n.get_noise_run_moar , [(loader, cluster, level) for r in range(NUMREPEATS)] )
    rr= s.execute()

    res= [process(level, 0) for level in rr]
    std= [processstd(level, 0) for level in rr]
    res2= [process(level, 1) for level in rr]
    std2= [processstd(level, 1) for level in rr]
    print(f"a={res}\nb={std}\n{rname}=[a,b,'RARI']\n\n")
    #print(f"a={res2}\nb={std2}\n{rname}_ari=[a,b,'ARI']\n\n")


myloaders=[l_3k]
lnames = ['k3']

for loader,lname in zip(myloaders, lnames):
    run(loader, f'{lname}_birch')
