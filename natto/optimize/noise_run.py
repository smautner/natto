import basics as ba 
from natto.input import load 
from natto.optimize import noise  as n 

#loader = lambda: load.load3k6k(subsample=500)  # lambda: load.loadarti("../data/art", 'si3', subsample= 1000)[0]

from functools import partial
loader = partial(load.load3k6k, subsample=500)

import natto.process as p 
from basics.sgexec import sgeexecuter as sge

cluster = partial(p.gmm_2, cov='full', nc = 8)




s=sge()
pool = 5
s.add_job( n.get_noise_run , [(loader,pool, cluster) for r in range(10)] )
rr= s.execute()[0]



#rr= ba.mpmap( n.get_noise_run , [(loader,pool, cluster ) for r in range(10)] , poolsize = 10, chunksize=1)



import numpy as np 
# split raris etc

rari = [[e[0] for e in r] for r in rr]
ari = [[e[1] for e in r] for r in rr]

print (rari)
print()
print()
print( np.array(rari).mean(axis=0))








