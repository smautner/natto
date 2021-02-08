import scanpy as sc
import pandas as pd
from lmz import *
import anndata as ad
import numpy as np


#### 
# oldest datasets
#######
load = lambda f: [l for l in open(f,'r').read().split('\n') if len(l)>1]


def do_subsample(adata, subsample, seed = None):
    if not subsample:
        return adata

    if subsample <1:
        sc.pp.subsample(adata, fraction=subsample, n_obs=None, random_state=seed, copy=False)
    else:
        if adata.shape[0] < subsample:
            return adata
        sc.pp.subsample(adata, fraction=None, n_obs=subsample, random_state=seed, copy=False)
    return adata

def loadlabels(labels, ids):
    cellid_to_clusterid = {row.split(',')[0]:hash(row.split(',')[1]) for row in labels[1:]} #
    clusterid_to_nuclusterid = {item:clusterid for clusterid, item in enumerate(sorted(list(set(cellid_to_clusterid.values()))))}
    #print (clusterid_to_nuclusterid)
    return np.array([ clusterid_to_nuclusterid.get(cellid_to_clusterid.get(idd[:-2],-1),-1)  for idd in ids])

def filter(adata, cells='mito'):

    if cells == 'seurat':
        adata = adata[adata.obs['labels'] != -1,:] # -1 are unannotated cells, there are quite a lot of these here :) 
    elif cells == 'mito':
        mito_genes = adata.var_names.str.startswith('MT-')
        adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
        adata = adata[adata.obs['percent_mito'] < 0.05, :]
        
    return adata

def load3k(cells: 'mito all seurat' ='mito', subsample=.15, seed = None)-> 'anndata object':
    adata =  sc.read_10x_mtx(
    '../data/3k/hg19/',  
    var_names='gene_symbols', cache=True)
    adata.obs['labels']= loadlabels(load( "../data/3k/pbmc.3k.labels"), load( "../data/3k/hg19/barcodes.tsv"))

    adata = filter(adata,cells)
    adata = do_subsample(adata, subsample,seed)
    return adata

def load6k(cells: 'mito all seurat' ='mito', subsample=.25, seed=None)-> 'anndata object':
    adata =  sc.read_10x_mtx(
    '../data/6k/hg19/',  
    var_names='gene_symbols', cache=True)

    adata.obs['labels']= loadlabels(load( "../data/6k/pbmc.6k.labels"), load( "../data/6k/hg19/barcodes.tsv"))

    adata = filter(adata,cells)
    adata = do_subsample(adata, subsample, seed)
    return adata



def loadpbmc(path=None, subsample=None, seed=None):
    adata = sc.read_10x_mtx( path,  var_names='gene_symbols', cache=True)
    adata = do_subsample(adata, subsample,seed)
    return adata

def load3k6k(subsample=False,seed=None):
    return load3k(subsample=subsample, seed=seed), load6k(subsample=subsample,seed=seed)

def loadp7de(subsample=False,pathprefix='..', seed=None):
    return loadpbmc('%s/data/p7d'%pathprefix ,subsample,seed), loadpbmc('%s/data/p7e'%pathprefix,subsample, seed)

def load4k8k(subsample=False,pathprefix='..',seed=None):
    return loadpbmc('%s/data/4k'% pathprefix,subsample, seed=seed), loadpbmc('%s/data/8k'%pathprefix,subsample,seed=seed)

def loadimmune(subsample=False, pathprefix='..',seed=None):
    return loadpbmc('%s/data/immune_stim/8'% pathprefix,subsample,seed=seed),\
           loadpbmc('%s/data/immune_stim/9'%pathprefix,subsample,seed=seed)


###
# grunreader
####

def loadgruen_single(path,subsample): 
    mtx_path = path+".1.counts_raw.csv.gz"
    things = pd.read_csv(mtx_path, sep='\t').T
    adata = ad.AnnData(things)

    truthpath = path+".cell_assignments.csv.gz"
    truth  = pd.read_csv(truthpath, sep='\t')
    #adata.obs['true']  = list(truth['assigned_cluster'])
    adata.obs['true']  = list(truth['celltype'])
    

    do_subsample(adata, subsample)
    return adata

def loadgruen(subsample=False, pathprefix='..', methods=['human1','human2']):
    return [loadgruen_single('%s/data/punk/%s'% (pathprefix,method),subsample) for method in methods]



########
# LOAD DCA 
#########

from anndata import read_h5ad

def loaddca_h5(path,subsample): 
    dca_path = path+"/adata.h5ad"
    adata = read_h5ad(dca_path)
    adata.obsm['tsv'] =  loaddca_late(path) 

    do_subsample(adata, subsample)

    return adata

def loaddca_late(path,subsample=None): 
    dca_path = path+"/latent.tsv"
    things = pd.read_csv(dca_path, index_col=0, header = None, sep='\t')
    return things
    '''
    adata = ad.AnnData(things)
    do_subsample(adata, subsample)
    return adata
    '''

#  h1  h2  h3  h4  immuneA  immuneB  m3k  m4k  m6k  m8k  p7d  p7e
def loaddca(fname, subsample = False, latent = False):
    path = "../data/dca/"+fname
    if latent:
        adata = loaddca_late(path, subsample)
    else:
        adata = loaddca_h5(path, subsample) 
    return adata

def loaddca_p7de(subsample):
    return loaddca("p7d", subsample), loaddca("p7e", subsample)
def loaddca_l_p7de(subsample):
    return loaddca("p7d", subsample, latent=True), loaddca("p7e", subsample,latent=True)

def loaddca_immuneab(subsample):
    return loaddca("immuneA", subsample), loaddca("immuneB", subsample)
def loaddca_l_immuneab(subsample):
    return loaddca("immuneA", subsample, latent= True), loaddca("immuneB", subsample, latent=True)

def loaddca_3k6k(subsample):
    return loaddca("m3k", subsample), loaddca("m6k", subsample)

def loaddca_l_3k6k(subsample):
    return loaddca("m3k", subsample, latent= True), loaddca("m6k", subsample, latent=True)


def loaddca_h1h2(subsample):
    return loaddca("h1", subsample), loaddca("h2", subsample)




###
# artificial data
#########

datanames = ["default","even",'si1','si2','c7','c6','c5','facl0.05','facl2']

def loadgalaxy(s=True,subsample=False):
    for i in range(1,10): # 1..9 ,,, 10 is below
        yield loadarti('../data/galaxy',f"fac{'s' if s else 'l'}0_{i}.",subsample=subsample)
    yield loadarti('../data/galaxy',f"fac{'s' if s else 'l'}1_0.",subsample=subsample)




def loadarti(path, dataname,subsample=False):

    batch = open(path+"/%sbatch.art.csv" % dataname, "r").readlines()[1:]
    batch = np.array([ int(x[-2]) for x in batch])
    
    cnts = open(path+"/%scounts.art.csv" % dataname, "r").readlines()[1:]
    
    cnts = [ Map(int,line.split(',')[1:])  for line in cnts]
    #cnts= sp.sparse.csr_matrix(Transpose(cnts))
    cnts= np.matrix(Transpose(cnts))
    
    a = ad.AnnData( cnts[batch == 1] )
    b = ad.AnnData( cnts[batch == 2] )
    b.var['gene_ids'] = {i:i for i in range(cnts.shape[1])}
    a.var['gene_ids'] = {i:i for i in range(cnts.shape[1])}
    a= do_subsample(a, subsample)
    b = do_subsample(b, subsample)

    return a,b
    

def loadarti_truth(path, p1, p2, dataname):
    grp = open(path+"/%sgroup.art.csv" % dataname, "r").readlines()[1:]
    grp = np.array([ int(x[-2]) for x in grp])
    
    batch = open(path+"/%sbatch.art.csv" % dataname, "r").readlines()[1:]
    batch = np.array([ int(x[-2]) for x in batch])
    
    
    gr1, gr2 =  grp[batch == 1 ], grp[batch==2]
    
    return gr1[p1], gr2[p2]
    
