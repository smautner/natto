
## install
```
# lapsolver needs cmake and this seems to work:
conda install cmake pip make gxx_linux-64 gcc_linux-64 fish
export CMAKE_MAKE_PROGRAM=$(which make)
export CC=$(which x86_64-conda_cos6-linux-gnu-gcc)
export CXX=$(which x86_64-conda_cos6-linux-gnu-g++)
# rari is not on pypi:
pip3 install git+https://github.com/nredell/rari                                 

pip3 install git+https://github.com/smautner/natto.git                          
```




## running natto 
```python 

import natto as na

# input is a pair of scanpy-anndata objects
# 'natto.input.load' constructs anndata objects from multiple formats
adata_1, adata_2 = na.l.loadimmune(subsample=250)


# preprocessing and clustering
data = na.prepare(adata_1,adata_2)


# similarity measure
print("similarity:",na.similarity(data))


# EM algorithm  and plotting
na.tunnelclust(data)
na.drawpair(data, tunnellabels = True)
```
