#!/home/mautner/.myconda/miniconda3/bin/python
import sys
import basics as ba
import numpy as np
from natto.input import load
from natto.input import preprocessing
from sklearn.metrics import pairwise_distances, rand_score
import ubergauss as ug 

dnames = "human1 human2 human3 human4 smartseq2 celseq2 celseq".split()

def get_score(name, name2):
    d1 = load.loadgruen_single(f"../data/punk/{name}", subsample=1000)
    d2 = load.loadgruen_single(f"../data/punk/{name2}", subsample=1000)
    pp = preprocessing.Data().fit(d1,d2,
                    debug_ftsel=False,
                    scale=True, 
                    maxgenes = 800)  
    allteX =  np.vstack(pp.dx)
    #labels = natto.process.gmm_1(allteX)
    labels = ug.get_model(allteX, poolsize=1,nclust_min=7, nclust_max=17).predict(allteX)
    real_labels = [i for d in [pp.a, pp.b] for i in d.obs['true'].values]
    rands = rand_score(real_labels, labels)
    return rands

if __name__ == "__main__":
    task = int(sys.argv[1])
    mtx = [[0.0, 0.7171103003533041, 0.6782515978674881, 0.5711449978177814, 0.7751568580958308, 0.8279503639443323, 0.8073698867678036], [0.7171103003533041, 0.0, 0.6458226579358887, 0.5870605129913772, 0.6700045301355889, 0.7311682549812814, 0.7832925704664048], [0.6782515978674881, 0.6458226579358887, 0.0, 0.5919020370236961, 0.7309107461572747, 0.8156476626788953, 0.7536937411088679], [0.5711449978177814, 0.5870605129913772, 0.5919020370236961, 0.0, 0.7017388744339382, 0.7656206989999302, 0.744133557048792], [0.7751568580958308, 0.6700045301355889, 0.7309107461572747, 0.7017388744339382, 0.0, 0.6868481130021408, 0.7669906186334121], [0.8279503639443323, 0.7311682549812814, 0.8156476626788953, 0.7656206989999302, 0.6868481130021408, 0.0, 0.7609737420476113], [0.8073698867678036, 0.7832925704664048, 0.7536937411088679, 0.744133557048792, 0.7669906186334121, 0.7609737420476113, 0.0]]
    mtx = np.array(mtx)
    morder = np.argsort(mtx[task])
    result = []
    for did in morder[1:]: 
        self = dnames[morder[0]]
        other = dnames[did]
        result.append( get_score(self, other))
    ba.dumpfile(result,"res/"+sys.argv[1]+"_"+sys.argv[2])


def res(indices,r): 
    for i in range(indices):
        indexrepeats =  np.array([ba.loadfile(f"res/{i}_{rep}") for rep in range(r) ]) 
        print ( indexrepeats.mean(axis=0))
