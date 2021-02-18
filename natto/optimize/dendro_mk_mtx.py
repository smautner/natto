#!/home/ubuntu/.myconda/miniconda3/bin/python
import sys
import basics as ba
import gc
import numpy as np

dnames = """Testis_10xchromium_SRA645804-SRS2823404_4197.h5  Testis_10xchromium_SRA645804-SRS2823409_4791.h5
Testis_10xchromium_SRA645804-SRS2823405_3598.h5  Testis_10xchromium_SRA645804-SRS2823410_4045.h5
Testis_10xchromium_SRA645804-SRS2823406_3989.h5  Testis_10xchromium_SRA645804-SRS2823412_5299.h5
Testis_10xchromium_SRA645804-SRS2823407_4046.h5  Testis_10xchromium_SRA645804-SRS3572594_4574.h5
Testis_10xchromium_SRA645804-SRS2823408_4306.h5"""
dnames = [d[:-3] for d in dnames.split()]

from natto.input.preprocessing import Data
from natto.out.quality import rari_score
from natto.input import load 
from natto import process
debug = False
if debug: 
    print(f"dnames:{len(dnames)}")

def similarity(stra, strb, rep): 
    scale = False, 
    subsample = 200 if debug else 1000 
    path='../data'
    d = Data().fit(load.load100(stra,path=path, subsample= subsample, seed = rep),
                 load.load100(strb, path=path, subsample= subsample, seed= rep), 
                debug_ftsel = False,
                scale= scale, 
                quiet = True, 
                pca = 20, 
                titles= ("a",'b'),
                make_even=True
            )
    l=process.gmm_2(*d.dx,nc=15, cov='full')
    return rari_score(*l, *d.dx)

def similarity2(a,b, r):
    res= similarity(a,b, r)
    gc.collect()
    return res

if __name__ == "__main__":
    print('argv',sys.argv)
    #task = int(sys.argv[1])
    #t2 =   int(sys.argv[2])
    #rep = int(sys.argv[3])
    task,t2,rep = map(int, sys.argv[1].strip().split(' '))
    #dnames = load.get100names(path='../data')

    home = dnames[task] 
    other = dnames[t2]
    result =  similarity2(home, other,rep)
    print (result)
    ba.dumpfile(result,"res/"+sys.argv[1].replace(" ",'_'))

def res(indices,r): 
    re = [] 
    for i in range(indices):
        indexrepeats =  np.array([ba.loadfile(f"res/{i}_{rep}") for rep in range(r) ]) 
        re.append( list(indexrepeats.mean(axis= 0))) 
    print(re) 


# for i in (seq 0 8); for j in (seq $i 8) ; for rep in (seq 0 20); echo $i $j $rep ;end;end;end | parallel -j 8 --bar ./dendro_mk_mtx.py

