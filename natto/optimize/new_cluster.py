

# %%  we compare the new cluster algorithms and find naisu

%load_ext autoreload
%autoreload 2


#%%
# LOAD DATA => [anndata]

import natto as na

data_raw = na.input.load.load3k6k(subsample=1000)


# %%
# preprocess
data = na.input.preprocessing.Data().fit(*data_raw, scale=False)



#%%
# cluster
from natto.process import *
from natto.out import draw

# default pref is median of similarity.
draw.cmp2(*afprop_2(*data.dx, damping=.75, preference =-130), *data.d2,title=['affinity prop',''])


#%%

draw.cmp2(*birch_2(*data.dx, n_clusters = 8), *data.d2,title=['birch',''])
draw.cmp2(*kmeans_2(*data.dx, nc = 8), *data.d2  ,title=['kmeans',''] )
# weird
draw.cmp2(*dbscan_2(*data.dx, min_samples = 10, eps = .5), *data.d2, title=['DBSCAN',''])

# doesnt seem to work at all
draw.cmp2(*optics_2(*data.dx, min_samples=10, max_eps = 100, p=2, xi=.01), *data.d2, title=['optics',''])

from sklearn.cluster import estimate_bandwidth as ebw
draw.cmp2(*meansh_2(*data.dx,bandwidth = ebw(data.dx[0], quantile = .1)), *data.d2,title=['meanshift',''])


draw.cmp2(*spec_2(*data.dx, nc = 8), *data.d2,title=['spectral',''])

# %%
