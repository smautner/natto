import matplotlib.pyplot as plt
from natto.util import dataloader as data
from natto.util import draw
import numpy as np
import pandas as pd
import scanpy as sc
import umap
import warnings
warnings.filterwarnings('ignore')

from natto.cluster import markers 
import umap
from natto import hungutil as h
from itertools import chain, combinations 
from natto.util.quality import gausssim



loaders = [data.loadp7de,data.load4k8k,data.loadimmune]
titles = [('p7d','p7e'),('4k','8k'),('immuneA','immuneB')]
loaders = [data.loadimmune]
titles = [('immuneA','immuneB')]
debug = False
numcells = 1000
algo = 'seurat' # select seurat or gmm 

## PREPROCESSING BY SEURAT 
from natto.cluster.seurat import reseurat
import anndata as ad
if algo=='seurat':
    for loader, (t1, t2) in zip(loaders, titles):
        r = reseurat()
        set1,set2 = loader(subsample=numcells,pathprefix='/home/ikea/HungarianClustering')
        a, red = r.soy(set1)
        b, _  = r.soy(set2)
        a,ca = a.obsm['X_pca'],a.obs['louvain']
        b,cb = b.obsm['X_pca'],b.obs['louvain']
        
        y_combined = markers.sim.predictlou(None,np.vstack((a,b)),{"n_neighbors":10})
        
        Yh,Y2h, _= h.bit_by_bit(a,b,ca,cb, debug=debug,normalize=True,maxerror=.10,
				showset=set(),
                                saveheatmap=None)
        labelappend = { k: "%.2f" % v  for k,v in gausssim(a,b,Yh,Y2h).items()}



if algo == 'gmm':
    for stuff in ['ttest']:
        for numgenes in [5]:
            for loader, (t1, t2) in zip(loaders, titles):
                m = markers.markers(*loader(subsample=numcells, pathprefix= "/home/ikea/HungarianClustering"))
                a,b,ca,cb = m.process("/home/ikea/HungarianClustering/data/%s_%s_topgenes.csv" % (t1,stuff ),
                                      "/home/ikea/HungarianClustering/data/%s_%s_topgenes.csv" %(t2, stuff),
                                      numgenes,clust='gmm', dimensions=6)
                red = umap.UMAP()
                red.fit(np.vstack((a,b)))
                y_combined = markers.sim.predictgmm(max(len(np.unique(cb)),len(np.unique(ca))),m.mymap.transform(np.vstack((a,b))))
                Yh,Y2h,_ = h.bit_by_bit(a,b,ca,cb, debug=debug,normalize=True,maxerror=.10,
                                        #showset = {'drawdist','heatmap','table','table_latex','renaming'},
                                        #showset = {'sankey','heatmap','table','table_latex','renaming'},
                                        showset =set(),
                                        saveheatmap=None)
                labelappend = { k: "%.2f" % v  for k,v in gausssim(a,b,Yh,Y2h).items()}
                
