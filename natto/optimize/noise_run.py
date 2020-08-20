import basics as ba 
from natto.input import load 
from natto.optimize import noise  as n 
import numpy as np 
from functools import partial
import natto.process as p 
from basics.sgexec import sgeexecuter as sge





cluster = partial(p.gmm_2, cov='full', nc = 15)
cluster = partial(p.leiden_2,resolution=.5)
loader = partial(load.loadgruen_single, path = '../data/punk/human3',  subsample=1500)
loader = partial(load.load3k, subsample=1500)


s=sge()
for level in range(0,110,10):
    s.add_job( n.get_noise_run_moar , [(loader, cluster, level) for r in range(50)] )
rr= s.execute()




'''
pool=10
rr= ba.mpmap( n.get_noise_run , [(loader,pool, cluster ) for r in range(10)] , poolsize = 10, chunksize=1)
'''

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
print ([processstd(level, 0) for level in rr])
print ([processstd(level, 1) for level in rr])

