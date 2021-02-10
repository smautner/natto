#!/home/mautner/.myconda/miniconda3/bin/python
import os 
import sys
import basics as ba
import numpy as np

from natto.input import load

if __name__ == "__main__":
    names =  load.get100names()
    task = int(sys.argv[1])

    myname = names[task]
    fname = f"../data/100/data/{myname}.h5"
    doit =  os.path.exists(fname)

    if doit:
        # z= load.get100(myname) LOAD GZIP STUFF
        # ad.write(fname, compression='gzip')
        load.load100addtruth(myname)


    #ba.dumpfile(result,"res/"+sys.argv[1]+"_"+sys.argv[2])

def res(indices,r): 
    print("meganaisu")
    #for i in range(indices):
    #    indexrepeats =  np.array([ba.loadfile(f"res/{i}_{rep}") for rep in range(r) ]) 
    #    print ( indexrepeats.mean())
