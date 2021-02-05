

# script calculates clustersim for one row of the dendro-simmilarity matrix
# ill have to write a script for convenient qsubbing... 

import sys
import numpy as np
row =  sys.argv[1] 

# determine comparison order
mtx = [[0.0, 0.7171103003533041, 0.6782515978674881, 0.5711449978177814, 0.7751568580958308, 0.8279503639443323, 0.8073698867678036], [0.7171103003533041, 0.0, 0.6458226579358887, 0.5870605129913772, 0.6700045301355889, 0.7311682549812814, 0.7832925704664048], [0.6782515978674881, 0.6458226579358887, 0.0, 0.5919020370236961, 0.7309107461572747, 0.8156476626788953, 0.7536937411088679], [0.5711449978177814, 0.5870605129913772, 0.5919020370236961, 0.0, 0.7017388744339382, 0.7656206989999302, 0.744133557048792], [0.7751568580958308, 0.6700045301355889, 0.7309107461572747, 0.7017388744339382, 0.0, 0.6868481130021408, 0.7669906186334121], [0.8279503639443323, 0.7311682549812814, 0.8156476626788953, 0.7656206989999302, 0.6868481130021408, 0.0, 0.7609737420476113], [0.8073698867678036, 0.7832925704664048, 0.7536937411088679, 0.744133557048792, 0.7669906186334121, 0.7609737420476113, 0.0]]
mtx = np.array(mtx)
startinstance = row
morder = np.argsort(mtx[startinstance]) 


from natto.input import load
dnames = "human1 human2 human3 human4 smartseq2 celseq2 celseq".split()
import natto
from natto.input import preprocessing
from sklearn.metrics import pairwise_distances, rand_score
import functools

start = dnames[morder[0]]
randscr=[]
REPEATS = 2
from basics import sgexec 
SGE = sgexec.sgeexecuter()


def get_score(dat):
    dat = [d() for d in dat]
    pp = preprocessing.Data().fit2(*dat,
                    debug_ftsel=False,
                    scale=True, 
                    maxgenes = 800)  
    allteX =  np.vstack(pp.dx)
    labels = natto.process.gmm_1(allteX)
    real_labels = [i for d in [pp.a, pp.b] for i in d.obs['true'].values]
    rands = rand_score(real_labels, labels)
    return rands


for did in morder[1:]: 
    other = dnames[did]
    dat = [ functools.partial(load.loadgruen_single, f"../data/punk/{dname}",subsample=1000)  for dname in [start, other]]
    SGE.add_job( get_score, [dat]*REPEATS)


SGE.execute()
rr = SGE.collect()
for r in rr:
    mean = np.array(r).mean()
    print (mean)