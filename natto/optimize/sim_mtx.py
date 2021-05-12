#!/home/ubuntu/.myconda/miniconda3/bin/python


import sys
import os
writeto = "res/"+sys.argv[1].replace(" ",'_')

#if os.path.exists(writeto):
#    print("file exists")
#    exit()

import basics as ba
import gc
import numpy as np
'''
dnames = load.get100names(path='../data')
'''
dnames = """Testis_10xchromium_SRA645804-SRS2823404_4197.h5  Testis_10xchromium_SRA645804-SRS2823409_4791.h5
Testis_10xchromium_SRA645804-SRS2823405_3598.h5  Testis_10xchromium_SRA645804-SRS2823410_4045.h5
Testis_10xchromium_SRA645804-SRS2823406_3989.h5  Testis_10xchromium_SRA645804-SRS2823412_5299.h5
Testis_10xchromium_SRA645804-SRS2823407_4046.h5  Testis_10xchromium_SRA645804-SRS3572594_4574.h5
Testis_10xchromium_SRA645804-SRS2823408_4306.h5"""
dnames = [d[:-3] for d in dnames.split()]

from natto.process import Data
from natto.out.quality import rari_score
from natto import input
from natto import process
from natto.process.cluster import gmm_2


dnames = input.get100names(path='../data')
debug = False
if debug: 
    print(f"dnames:{len(dnames)}")

def similarity(stra, strb, rep): 
    scale = False, 
    subsample = 200 if debug else 2500 
    path='../data'
    seed1, seed2 = rep,rep
    if stra == strb:
        seed2 = 29347234
    d = Data().fit([input.load100(stra,path=path, subsample= subsample, seed = seed1),
                 input.load100(strb, path=path, subsample= subsample, seed= seed2)], 
                visual_ftsel = False,
                scale= scale, 
                pca = 20, 
                umaps=[10],
                make_even=True # adjusted to new preproc but untested, sortfield default -1 might be a problem
            )
    print("clustering..",end='')
    l=gmm_2(*d.d10,nc=15, cov='full')
    r=rari_score(*l, *d.d10)
    return r

def similarity_gene(stra, strb, rep): 
    '''
    for comparison we also see how much gene overlap there is
    '''
    scale = False, 
    subsample = 200 if debug else 1000
    path='../data'
    seed1, seed2 = rep,rep
    if stra == strb:
        seed2 = 29347234
    d = Data().fit([input.load100(stra,path=path, subsample= subsample, seed = seed1),
                 input.load100(strb, path=path, subsample= subsample, seed= seed2)], 
                visual_ftsel = False,
                scale= scale, 
                pca = 0, 
                umaps=[],
                make_even=True # adjusted to new preproc but untested, sortfield default -1 might be a problem
            )
    
    return sum([ a and b for a,b in zip(*d.genes)])



if __name__ == "__main__":
    task,t2,rep = map(int, sys.argv[1].strip().split(' '))
    home = dnames[task] 
    other = dnames[t2]
    if debug: print("fanmes", home, other)
    result =  similarity_gene(home, other,rep)
        
    print(result)
    ba.dumpfile(result,writeto)

'''
source setvar.fish 
    for i in (seq 0 99)
    for j in (seq $i 99)
    for rep in (seq 0 10)
    if ! test -e ./res/(string join _ $i $j $rep) && echo "$i $j $rep"; end
    end; end; end |  parallel -j 32 --bar ./sim_mtx.py
'''
