#!/home/ubuntu/.myconda/miniconda3/bin/python

import sys
import basics as ba
import numpy as np
from natto import input as load
from natto import process
from sklearn.metrics import pairwise_distances, adjusted_rand_score
import ubergauss as ug 
import natto
from natto.process import cluster as cluster
from natto.process.cluster.k2means import tunnelclust 
from natto.out.quality import rari_srt

debug = False
sampnum = 200 if debug else 400 # when using 1000  there is not much difference.. 




def score(X,y): 
    labels = cluster.gmm_1(X,nc=15)
    rari = rari_srt(labels,y,X,X)
    return adjusted_rand_score(labels, y), rari



def get_score(n1, n2, seed):
    

    # PREPROCESS:
    loader = lambda x, seed : load.load100(x,subsample=sampnum,path='../data', remove_unlabeled = True, seed = seed)
    d1 = loader(n1,seed)
    if n1==n2:
        seed += 1982369162
    d2 = loader(n2,seed)
    pp = process.Data().fit([d1,d2],
                    visual_ftsel=False,
                    scale=True, 
                    umaps =[10],
                    sortfield = 1, # sorting for tunnelclust 
                    selectgenes = 800)  

    
    trve = lambda pp: [i for d in pp.data for i in d.obs['true'].values]
    X = np.vstack(pp.d10)

    joint_score, jrari = score(X, trve(pp))
    ascore, arari = score(pp.d10[0],list(pp.data[0].obs['true']))
    bscore,brari = score(pp.d10[1],list(pp.data[1].obs['true']))

    mylabels = tunnelclust(*pp.d10)
    mylabels = np.hstack(mylabels)
    nattoclust_score = adjusted_rand_score( mylabels , trve(pp)) 
    nattoclust_rari = rari_srt( mylabels , trve(pp),X,X )

    return joint_score,jrari, ascore,arari, bscore,brari, nattoclust_score, nattoclust_rari



def loadtasks():
    return eval(open('/home/ubuntu/taskpairspp3','r').read())


if __name__ == "__main__":
    #print("argv: ", sys.argv)
    task,rep = map(int, sys.argv[1].strip().split(' '))
    self, other = loadtasks()[task]
    result=  get_score(self, other, seed = rep)
    print("res: ", result)
    #ba.dumpfile(result,"res/"+sys.argv[1].replace(" ",'_'))
    #print("all good")

