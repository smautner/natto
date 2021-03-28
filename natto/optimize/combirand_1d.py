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


debug = True
sampnum = 200 if debug else 1000




def score(X,y): 
    labels = cluster.gmm_1(X,nc=15)
    return adjusted_rand_score(labels, y)



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
                    selectgenes = 800)  

    
    trve = lambda pp: [i for d in pp.data for i in d.obs['true'].values]
    joint_score = score(np.vstack(pp.d10), trve(pp))
    
    ascore = score(pp.d10[0],list(pp.data[0].obs['true']))
    bscore = score(pp.d10[1],list(pp.data[1].obs['true']))
    
    return joint_score, ascore, bscore



def loadtasks():
    return eval(open('/home/ubuntu/taskpairspp3','r').read())


if __name__ == "__main__":
    #print("argv: ", sys.argv)
    task,rep = map(int, sys.argv[1].strip().split(' '))
    self, other = loadtasks()[task]
    result=  get_score(self, other, seed = rep)
    print("res: ", result)
    ba.dumpfile(result,"res/"+sys.argv[1].replace(" ",'_'))
    #print("all good")

