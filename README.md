
## install
```
install python3-pip and cmake

pip3 install git+https://github.com/nredell/rari                                 

pip3 install git+https://github.com/smautner/natto.git                          
```


## install problems:
```

# scanpy seems to  be bugged 
pip3 install --upgrade  scanpy==1.4.4.post1
remove anndata
pip3 install --upgrade  anndata==0.6.22.post1
# -> this may be false, i think i need to build a cache for each dataset first,
# multiple proicesses trying to build a cache at the same time fails...


# installing lapsolver on VERY problematic machines
1. clone lapsolver
2. give cmake the  paths to c++ and cc somewhere in the config
3. remove crashing lines from setup.py
```



## running natto 
```python 
# 1. input is a pair of scanpy-anndata objects

from natto.input import load
adata_1, adata_2 = load.loadimmune(subsample=1000)
# 'load' constructs anndata objects from multiple formats


# 2. preprocessing and clustering

from natto.input import preprocessing
data = preprocessing.Data().fit(adata_1,adata_2, quiet=True, debug_ftsel =False)

import natto.process as p
labels = p.gmm_2(*data.dx, nc=10)


# 3. distance
from natto.out.quality import rari_score as score
rari, rand_index  = score(*labels, *data.dx)
print(f"rari: {rari}")


# 4. EM algorithm
from natto.process import k2means
data.sort_cells()
labels,outliers = k2means.simulclust(*data.dx,labels[0])
labels[outliers] = -1

#import natto.out.draw as draw
#draw.cmp2(labels,labels,*data.d2)
```
