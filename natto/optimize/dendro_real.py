
#%%
from matplotlib import pyplot as plt
%load_ext autoreload
%autoreload 2

#%%
from natto.input import load
import numpy as np
from natto.out.quality import make_rari_compatible
dnames = "human1 human2 human3 human4 smartseq2 celseq2 celseq".split()

#dat = load.loadgruen_single( f"../data/punk/{dnames[0]}",subsample=600) 

# %%
# are theese in the right order? mmm anyways should be the real labels

# %%
# we need somtghing like this later,,,,
import natto


    
#%% lets organize a matrix and the first instance 
mtx = [[0.0, 0.7171103003533041, 0.6782515978674881, 0.5711449978177814, 0.7751568580958308, 0.8279503639443323, 0.8073698867678036], [0.7171103003533041, 0.0, 0.6458226579358887, 0.5870605129913772, 0.6700045301355889, 0.7311682549812814, 0.7832925704664048], [0.6782515978674881, 0.6458226579358887, 0.0, 0.5919020370236961, 0.7309107461572747, 0.8156476626788953, 0.7536937411088679], [0.5711449978177814, 0.5870605129913772, 0.5919020370236961, 0.0, 0.7017388744339382, 0.7656206989999302, 0.744133557048792], [0.7751568580958308, 0.6700045301355889, 0.7309107461572747, 0.7017388744339382, 0.0, 0.6868481130021408, 0.7669906186334121], [0.8279503639443323, 0.7311682549812814, 0.8156476626788953, 0.7656206989999302, 0.6868481130021408, 0.0, 0.7609737420476113], [0.8073698867678036, 0.7832925704664048, 0.7536937411088679, 0.744133557048792, 0.7669906186334121, 0.7609737420476113, 0.0]]
mtx = np.array(mtx)
print(mtx.shape)
su = mtx.sum(axis=0)
order = np.argsort(su)
startinstance = order[0]
morder = np.argsort(mtx[startinstance]) 

#%%

import natto
from natto.input import pp_many, preprocessing
from sklearn.metrics import pairwise_distances, rand_score


start = dnames[morder[0]]
randscr=[]
for did in morder[1:]: 
    other = dnames[did]
    
    # ok now we mix these 2 and calculate the rand
    #dat = [load.loadgruen_single( f"../data/punk/{dname}",subsample=700)  for dname in currentnames]
    #pp = pp_many.Data().fit(dat,debug_ftsel=False,scale=True, maxgenes = int(1500/len(currentnames))) # TODO: maxgenes for all parts.. together
    r=[]
    for i in range(5):
        dat = [load.loadgruen_single( f"../data/punk/{dname}",subsample=1000)  for dname in [start, other]]
        pp = preprocessing.Data().fit(*dat,
                    debug_ftsel=False,
                    scale=True, 
                    maxgenes = 800)  
        allteX =  np.vstack(pp.dx)
        labels = natto.process.gmm_1(allteX)
        real_labels = [i for d in [pp.a, pp.b] for i in d.obs['true'].values]
        rands = rand_score(real_labels, labels)
        r.append(rands)

    randscr.append(np.array(r).mean())
    print ("RAND", randscr) 

# then do this starting with any 
#  [0.8924611305652826, 0.9637320660330164, 0.9003336668334168, 0.7692594297148575, 0.7688283141570785, 0.7588649324662331]
# this is when we dont subsample for the samples to be even: 
# 0.8940090045022512, 0.9481543771885942, 0.8994602301150575, 0.7643338669334667, 0.76911995997999, 0.7652657328664332

# 2. optimization of clustering algorithms on real labels? 
    # the best algo might not yield the best curve on the noise plot
    # should i just plot 999 noise plots? .... 

# %%

plt.plot([0.0, 0.3300936679537677, 0.5463896728205502, 0.5174288042268417, 0.592070366870298, 0.7271271170399566, 0.620300052155158, 0.8274519149942243, 1.1065286330261679, 1.3148192829009209, 0.8152652490241052, 0.7973065804065462, 0.9301881753359096, 0.45295444058709705, 0.32567484884599446, -1.403693945693437e-15])


pts = np.array([(0, -1827.3906456669301), (1, -2147.049060230715), (2, -2404.2240822687013), (3, -2620.138165444961), (4, -2816.9289258875756), (5, -2985.2552047698364), (6, -3154.5045790035438), (7, -3377.1395240046722), (8, -3492.049711677463), (9, -3696.9222009125187), (10, -3669.3474583552197), (11, -3958.5636085227397), (12, -4013.3850893322033), (13, -4140.280414848434), (14, -4144.545630718626), (15, -4406.99120361149)])

plt.show()
#plt.plot(pts[:,0], pts[:,1])
plt.plot( pts[:,1])

# %%

trash_results ='''[[0.61747901 0.62203223 0.64039477 0.7056479  0.6442787  0.67223992] 
[0.64174117 0.60243936 0.57053833 0.63765863 0.58666886 0.57881819]  
[0.58956466 0.61180228 0.63098447 0.5658358  0.5520149  0.58803452]  
[0.63342867 0.61952885 0.59060091 0.62122232 0.60385168 0.6320553 ]  
[0.57402528 0.59613174 0.63044296 0.56261741 0.54436771 0.68763476]  
[0.59870067 0.58021905 0.60931588 0.625537   0.58815912 0.67873879]  
[0.5963387  0.55208511 0.61367034 0.5225367  0.58184223 0.65969361]]'''
trash = []
from lmz import *
for line in trash_results.split('\n'):
    line = line.strip()
    line=line.replace('[','')
    line=line.replace(']','')
    trash.append(np.array(Map(float,[thing for thing in line.split(' ') if thing])))
print(trash)

for startname, reddist, mydist in zip(dnames,trash, mtx):
    mydis = np.sort(mydist)[1:]
    print(mydis, reddist)
    plt.plot(mydis, reddist)
    plt.title(startname)
    plt.xlabel('rari')
    plt.ylabel('accuracy when paired with neighbor')
    plt.show()
# %%
