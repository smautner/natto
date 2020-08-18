import basics as ba 
from natto.input import load 
from natto.optimize import noise  as n 
import numpy as np 

#loader = lambda: load.load3k6k(subsample=500)  # lambda: load.loadarti("../data/art", 'si3', subsample= 1000)[0]

from functools import partial
loader = partial(load.load3k, subsample=1500)

import natto.process as p 
from basics.sgexec import sgeexecuter as sge

cluster = partial(p.gmm_2, cov='full', nc = 8)



s=sge()
for level in range(0,110,10):
    s.add_job( n.get_noise_run_moar , [(loader, cluster, level) for r in range(100)] )
rr= s.execute()




'''
pool=10
rr= ba.mpmap( n.get_noise_run , [(loader,pool, cluster ) for r in range(10)] , poolsize = 10, chunksize=1)
'''

print(rr)
#

def process(level, c):
    l =np.array(level)
    return l.mean(axis = 0 )[c]

def processVar(level, c):
    l =np.array(level)
    return l.var(axis = 0 )[c]

def processstd(level, c):
    l =np.array(level)
    return l.std(axis = 0 )[c]


print ([process(level, 0) for level in rr])
print ([process(level, 1) for level in rr])
print ([processVar(level, 0) for level in rr])
print ([processVar(level, 1) for level in rr])
print ([processstd(level, 0) for level in rr])
print ([processstd(level, 1) for level in rr])



