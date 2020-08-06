
   ( install python3-pip cmake) 
   pip3 install git+https://github.com/nredell/rari                                 
   pip3 install git+https://github.com/smautner/natto.git                          
   pip3 install git+https://github.com/smautner/ubergauss.git  
    
   # scanpy seems to be bugged
   pip3 install --upgrade  scanpy==1.4.4.post1
   remove anndata
   pip3 install --upgrade  anndata==0.6.22.post1
