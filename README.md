
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
