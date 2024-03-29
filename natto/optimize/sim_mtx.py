#!/home/ubuntu/.myconda/miniconda3/bin/python


import sys
import os
writeto = sys.argv[1]+"/"+'_'.join(sys.argv[2:])
argz = sys.argv[2:]
if int(argz[0]) < int(argz [1]):
    exit()

import matplotlib
#writeto = "res/"+sys.argv[1].replace(" ",'_')

#if os.path.exists(writeto):
#    print("file exists")
#    exit()

import basics as ba
import gc
import numpy as np
'''
dnames = load.get100names(path='../data')
dnames = input.get100names(path='../data')
'''
dnames = """Testis_10xchromium_SRA645804-SRS2823404_4197.h5  Testis_10xchromium_SRA645804-SRS2823409_4791.h5
Testis_10xchromium_SRA645804-SRS2823405_3598.h5  Testis_10xchromium_SRA645804-SRS2823410_4045.h5
Testis_10xchromium_SRA645804-SRS2823406_3989.h5  Testis_10xchromium_SRA645804-SRS2823412_5299.h5
Testis_10xchromium_SRA645804-SRS2823407_4046.h5  Testis_10xchromium_SRA645804-SRS3572594_4574.h5
Testis_10xchromium_SRA645804-SRS2823408_4306.h5"""
dnames = [d[:-3] for d in dnames.split()]

from natto.process import Data, Data_DELME
from natto import input
from natto import process
from natto.process.cluster import gmm_2, spec_2, kmeans_2
from natto.out.quality import rari_score
from umap import UMAP
from natto.optimize.util import sim as neighborssim


#dnames = input.get100names(path='../data')
dnames = input.get71names()
debug = False
debugOUT = False
if debug:
    print(f"dnames:{len(dnames)}")

def similarity(stra, strb, rep):
    scale = False,
    subsample = 200 if debug else 1500
    path='../data'
    seed1, seed2 = rep,rep
    if stra == strb:
        seed2 += 2937234
    scelldata = Data().fit([input.load100(stra,path=path, subsample= subsample, seed = seed1),
                 input.load100(strb, path=path, subsample= subsample, seed= seed2)],
                visual_ftsel = False,
                scale= scale,
                pca = 20,
                umaps=[2,10] if debugOUT else [10] ,
                make_even=True # adjusted to new preproc but untested, sortfield default -1 might be a problem
            )

    import anndata as ad
    rr = ad.concat(scelldata.data)

    rr.obs_names_make_unique()
    #breakpoint()
    rr.X= rr.X.todense()
    rr.write_h5ad("exporttest.h5ad")




    #print("clustering..",end='')

    # cluster

    #specdim = 15
    #spectral_labels=spec_2(scelldata.PCA[0][:,:specdim], scelldata.PCA[1][:,:specdim],nc=15)
    # spectral_labels=spec_2(scelldata.d10[0][:,:specdim], scelldata.d10[1][:,:specdim],nc=15)


    # print(f" {np.unique(scelldata.d10[0],return_counts = True, axis=0)[1]}")
    # print(f" {np.unique(scelldata.d10[1],return_counts = True, axis=0)[1]}")
    # get score
    r=rari_score(*gmm_2(*scelldata.d10,nc=15, cov='full'), *scelldata.d10)
    r2=rari_score(*gmm_2(*scelldata.d10,nc=15, cov='tied'), *scelldata.d10)
    s=rari_score(*spec_2(*scelldata.d10,nc=15), *scelldata.d10)
    #s8=rari_score(*spec_2(*scelldata.d10,nc=8), *scelldata.d10)
    if debugOUT:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('module://matplotlib-sixel')
        plt.scatter(scelldata.d2[0][:,0],scelldata.d2[0][:,1],  alpha = .5)
        #plt.show()
        #plt.close()
        plt.scatter(scelldata.d2[1][:,0],scelldata.d2[1][:,1], alpha = .5)
        plt.show()
        plt.close()
    '''
    kmeans_labels=kmeans_2(*scelldata.d10,nc=15)
    import matplotlib.pyplot as plt
    from sklearn import manifold
    #zz = manifold.MDS().fit_transform(scelldata.d10[0])
    zz = UMAP().fit_transform(scelldata.d10[0])
    plt.scatter(zz[:,0],zz[:,1], c= spectral_labels[0])
    plt.show()
    plt.close()

    #zz = manifold.MDS().fit_transform(scelldata.d10[1])
    zz = UMAP().fit_transform(scelldata.d10[1])
    plt.scatter(zz[:,0],zz[:,1], c= spectral_labels[1])
    plt.show()
    print(f" {zz.shape} {np.unique(zz,return_counts = True, axis=0)[1]}")
    '''

    ezcompe = neighborssim(*scelldata.d10)
    ezcompe2 = neighborssim(*scelldata.PCA)
    ezcompe2 = neighborssim(*scelldata.data) # !! RUN THIS :))))
    return r,r2,s,ezcompe2, ezcompe



def similarity_manygenes(stra, strb, rep):
    scale = False,
    subsample = 200 if debug else 1000
    path='../data'
    seed1, seed2 = rep,rep
    if stra == strb:
        seed2 = 29347234
    d = Data_DELME().fit([input.load100(stra,path=path, subsample= subsample, seed = seed1),
                 input.load100(strb, path=path, subsample= subsample, seed= seed2)],
                visual_ftsel = False,
                scale= scale,
                pca = 20,
                umaps=[10],
                make_even=True # adjusted to new preproc but untested, sortfield default -1 might be a problem
            )
    print("clustering..",end='')

    def scr(x):
        x=x[-1] # -1 should be the d10
        l=gmm_2(*x,nc=15, cov='full')
        return rari_score(*l, *x)
    lal = [ scr(x) for x in d.projections ]
    return lal




class myData(Data):
    def preprocess(self, selector, selectgenes, selectorargs):
        return super().preprocess(selector,selectgenes,selectorargs, savescores=True)

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
    d = myData().fit([input.load100(stra,path=path, subsample= subsample, seed = seed1),
                 input.load100(strb, path=path, subsample= subsample, seed= seed2)],
                visual_ftsel = False,
                scale= scale,
                pca = 0,
                umaps=[],
                make_even=True # adjusted to new preproc but untested, sortfield default -1 might be a problem
            )
    #return sum([ a and b for a,b in zip(*d.genes)])
    #return [countoverlap(*d.genescores,num) for num in range(50,1300,50)]
    return [countoverlap(*d.genescores,num) for num in range(50,2000,50)]

def countoverlap(a,b,num):
    srta = set(np.argsort(a)[:num])
    srtb = set(np.argsort(b)[:num])
    res = len(srta&srtb)
    return res


if __name__ == "__main__":
    matplotlib.use('module://matplotlib-sixel')
    print (argz)
    task,t2,rep = map(int, sys.argv[2:])
    #task,t2,rep = map(int, sys.argv[1].strip().split(' '))
    home = dnames[task]
    other = dnames[t2]
    if debug: print("fanmes", home, other)
    result =  similarity(home, other,rep)
    print(result)
    ba.dumpfile(result,writeto)
    sys.exit(2)

'''
source setvar.fish
    for i in (seq 0 99)
    for j in (seq $i 99)
    for rep in (seq 0 10)
    if ! test -e ./res/(string join _ $i $j $rep) && echo "$i $j $rep"; end
    end; end; end |  parallel -j 32 --bar ./sim_mtx.py
'''

