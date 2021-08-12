#!/home/ubuntu/.myconda/miniconda3/bin/python

from lmz import *
import sys
import basics as ba
import numpy as np
from natto import input as load
from natto import process
from sklearn.metrics import pairwise_distances, adjusted_rand_score, f1_score
import ubergauss as ug 
import natto
from natto.process import cluster as cluster
from natto.process.cluster.k2means import tunnelclust 
from natto.out.quality import rari_srt

debug = False
sampnum = 200 if debug else 2000 # when using 1000  there is not much difference.. 




def score(X,y): 
    labels = cluster.gmm_1(X,nc=15)
    rari = rari_srt(labels,y,X,X)
    return adjusted_rand_score(labels, y), rari


def nuscore(X,labels,y1,y2): 
    '''
    '''
    
    # split x and y 
    h1,h2 = labels[:len(y1)],  labels[len(y1):]
    x1,x2 = X[:len(y1),:],  X[len(y1):,:]
   
    # get scores seperately 
    rari = rari_srt(h1,y1,x1,x1) + rari_srt(y2,h2,x2,x2)
    ara = adjusted_rand_score(h1, y1)+ adjusted_rand_score(h2,y2)
    return ara/2, rari/2

def get_score(n1, n2, seed):
    '''
    '''
    # PREPROCESS:
    loader = lambda x, seed : load.load100(x,subsample=sampnum,path='../data', remove_unlabeled = True, seed = seed)
    d1 = loader(n1,seed)
    if n1==n2:
        seed += 1982369162
    d2 = loader(n2,seed)
    pp = process.Data().fit([d1,d2],
                    visual_ftsel=False,
                    scale=True, 
                    make_even = True,
                    umaps =[10],
                    sortfield = 1, # sorting for tunnelclust 
                    selectgenes = 800)  

    
    y1,y2 = [list(d.obs['true']) for d in pp.data]
    X = np.vstack(pp.d10)

    #ascore, arari = score(pp.d10[0],y1)
    #bscore,brari = score(pp.d10[1],y2)
    # joint cluster
    #labels = cluster.gmm_1(X,nc=15)
    #mylabels = tunnelclust(*pp.d10)
    #mylabels = np.hstack(mylabels)
    # score
    #joint_score, jrari = nuscore(X,labels, y1,y2)
    #nattoclust_score , nattoclust_rari = nuscore(X,mylabels,y1,y2)
    #otherstuff = score_smallfry(mylabels, labels, *pp.d10, y1,y2,nc)


    nc = max(map(len,map(np.unique,[y1,y2])))
    labels = cluster.gmm_1(X,nc=nc)
    h1,h2 = labels[:len(y1)],  labels[len(y1):]
    labels1 = cluster.gmm_1(pp.d10[0],nc=nc)
    labels2 = cluster.gmm_1(pp.d10[1],nc=nc)
    
    
    y1=np.array(y1)
    y2=np.array(y2)
    single=[]
    multi =[]
    sizes=[]
    for real,s,m in [[y1,labels1,h1],[y2,labels2,h2]]:
        for rlabel in np.unique(real):
            single.append(f1(rlabel,s,real))
            multi.append(f1(rlabel,m,real))
            sizes.append(sum(real==rlabel))
    return single, multi, sizes


def f1(l,m,r): # label to check, clustering, real_labels 
    rclust = max([ (lap(i,l,m,r),i) for i in np.unique(m)  ], key = lambda x:x[0])[1]
    return f1_score(r==l,m==rclust)

def lap(target, l,m,r):
    tmask = m==target
    rmask = r==l
    return sum(np.logical_and(tmask,rmask)) / sum(tmask)



'''
def score_smallfry(mylabels,labels,x1,x2,y1,y2,nc=15): 
    # get all the labels
    h1,h2 = labels[:len(y1)],  labels[len(y1):]
    j1,j2 = mylabels[:len(y1)],  mylabels[len(y1):]
    i1 = cluster.gmm_1(x1,nc=nc)
    i2 = cluster.gmm_1(x2,nc=nc)
    # ok lets start with ds1:
    sc = lambda x: list(score_clusters(y1,x,i1))
    s1 = sc(h1)
    s2 = sc(j1)
    sc = lambda x: list(score_clusters(y2,x,i2))
    s3 = sc(h2)
    s4 = sc(j2)
    return s1,s2,s3,s4
def score_clusters(y,yc,yhon): 
    #print ( Zip(y,yhon))
    for u in np.unique(y): 
        scom = count(y,u,yc)
        shon = count(y,u,yhon)
        #print(u, sum(y==u),shon) 
        yield sum(y==u), scom, shon 
def count(y,u,h):
    candidates = np.unique(h[y==u])
    def countmiss(thing):
        overlap = sum(np.logical_and(y==u,h==thing))
        return sum(y==u) - overlap  + sum(h==thing) - overlap
    return min(map(countmiss,candidates))
'''


def loadtasks():
    return eval(open('/home/ubuntu/taskpairspp3','r').read())


if __name__ == "__main__":
    #print("argv: ", sys.argv)
    task,rep = map(int, sys.argv[1].strip().split(' '))
    self, other = loadtasks()[task]
    result=  get_score(self, other, seed = rep)
    print("res: ", result)
    ba.dumpfile(result,"resOMG/"+sys.argv[1].replace(" ",'_'))
    #print("all good")

