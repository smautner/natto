import matplotlib.pyplot as plt
def draw():
    for s in p.d2:
        plt.scatter(s[:, 0], s[:, 1])
    plt.show()

import anndata as ad
import scanpy as sc
import natto.process as p

z = ad.read_h5ad("/home/ikea/data/3k.h5")
a= sc.pp.subsample(z,copy=True, n_obs =1000,random_state=1)
b= sc.pp.subsample(z,copy=True, n_obs =1000, random_state = 2)
c= sc.pp.subsample(z,copy=True, n_obs =1000, random_state = 3)



A = p.Data().fit([a.copy()],umaps=[2],visual_ftsel=False)
B = p.Data().fit([y.copy() for y in [a,b]],umaps=[2],visual_ftsel=False)
C = p.Data().fit([y.copy() for y in [a,b,c]],umaps=[2],visual_ftsel=False)
exit()
# draw(A) etc








