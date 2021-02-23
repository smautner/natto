#!/home/ubuntu/.myconda/miniconda3/bin/python
import os 
import sys
import basics as ba
import numpy as np
from natto.input import load



if __name__ == "__main__":
    # which file do we take care of?
    #names =  load.get100names(path='../data')
    #task = int(sys.argv[1])
    #myname = names[task]
    myname = sys.argv[1] 
    #myname = myname[:-10]
    print (myname)

    
    #fname = f"../data/{myname}.h5"
    #doit =  not os.path.exists(fname)

    z= load.get100gz(myname, path='../data') # LOAD GZIP STUFF
    print("LOADING FINE")
    #ad.write(fname, compression='gzip')
    load.load100addtruthAndWrite(z,myname,path='../data')


    #ba.dumpfile(result,"res/"+sys.argv[1]+"_"+sys.argv[2])

def res(indices,r): 
    print("meganaisu")
    #for i in range(indices):
    #    indexrepeats =  np.array([ba.loadfile(f"res/{i}_{rep}") for rep in range(r) ]) 
    #    print ( indexrepeats.mean())
