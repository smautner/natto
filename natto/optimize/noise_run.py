import basics as ba 
from natto.input import load 
from natto.optimize import noise  as n 

loader = lambda: load.loadarti("../data/art", 'si3', subsample= 1000)[0]

rr = ba.mpmap( n.get_noise_run , range(10), poolsize = -1 )

import numpy as np 
# split raris etc
rari = [[e[0] for e in r] for r in rr]
ari = [[e[1] for e in r] for r in rr]

print (rari)
print( np.array(rari).mean(axis=0))








