import scanpy as sc
from lmz import *
import anndata as ad
import numpy as np
import scipy as sp



load = lambda f: [l for l in open(f,'r').read().split('\n') if len(l)>1]

#load = lambda f: open(f,'r').read()


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

def load3k(cells: 'mito all seurat' ='mito', subsample=.15)-> 'anndata object':
    adata =  sc.read_10x_mtx(
    '../data/filtered_gene_bc_matrices/hg19/',  
    var_names='gene_symbols', cache=True)
    adata.obs['labels']= loadlabels(load( "../data/pbmc.3k.labels"), load( "../data/filtered_gene_bc_matrices/hg19/barcodes.tsv"))

    adata = filter(adata,cells)
    if subsample:
        if subsample <1:
            sc.pp.subsample(adata, fraction=subsample, n_obs=None, random_state=0, copy=False)
        else:
            sc.pp.subsample(adata, fraction=None, n_obs=subsample, random_state=0, copy=False)
    return adata

def load6k(cells: 'mito all seurat' ='mito', subsample=.25)-> 'anndata object':
    adata =  sc.read_10x_mtx(
    '../data/filtered_matrices_mex/hg19/',  
    var_names='gene_symbols', cache=True)

    adata.obs['labels']= loadlabels(load( "../data/pbmc.6k.labels"), load( "../data/filtered_matrices_mex/hg19/barcodes.tsv"))

    adata = filter(adata,cells)
    if subsample:
        if subsample <1:
            sc.pp.subsample(adata, fraction=subsample, n_obs=None, random_state=0, copy=False)
        else:
            sc.pp.subsample(adata, fraction=None, n_obs=subsample, random_state=0, copy=False)
    return adata


def loadpbmc(path, subsample=None):
    adata = sc.read_10x_mtx( path,  var_names='gene_symbols', cache=True)
    if subsample:
        if subsample <1:
            sc.pp.subsample(adata, fraction=subsample, n_obs=None, random_state=0, copy=False)
        else:
            sc.pp.subsample(adata, fraction=None, n_obs=subsample, random_state=0, copy=False)
    return adata

def load3k6k(subsample=False):
    return load3k(subsample=subsample), load6k(subsample=subsample)

def loadp7de(subsample=False,pathprefix='..'):
    return loadpbmc('%s/data/p7d'%pathprefix ,subsample), loadpbmc('%s/data/p7e'%pathprefix,subsample)

def load4k8k(subsample=False,pathprefix='..'):
    return loadpbmc('%s/data/4k'% pathprefix,subsample), loadpbmc('%s/data/8k'%pathprefix,subsample)

def loadimmune(subsample=False, pathprefix='..'):
    return loadpbmc('%s/data/immune_stim/8'% pathprefix,subsample), loadpbmc('%s/data/immune_stim/9'%pathprefix,subsample)


def loadarti(path):
    

    grp = open(path+"/group.csv", "r").readlines()[1:]
    grp = np.array( [ int(x[-2]) for x in grp] )
    
    batch = open(path+"/batch.csv", "r").readlines()[1:]
    batch = np.array([ int(x[-2]) for x in batch])
    
    cnts = open(path+"/counts.csv", "r").readlines()[1:]
    
    cnts = [ Map(int,line.split(',')[1:])  for line in cnts]
    cnts= sp.sparse.csr_matrix(Transpose(cnts))
    #cnts= np.matrix(Transpose(cnts))
    
    a = ad.AnnData( cnts[batch == 1] )
    b = ad.AnnData( cnts[batch == 2] )
    b.var['gene_ids'] = {i:i for i in range(cnts.shape[1])}
    a.var['gene_ids'] = {i:i for i in range(cnts.shape[1])}
    
    return a,b
    

def loadarti_truth(path):
    grp = open(path+"/group.csv", "r").readlines()[1:]
    grp = [ int(x[-2]) for x in grp]
    
    batch = open(path+"/batch.csv", "r").readlines()[1:]
    batch = [ int(x[-2]) for x in batch]
    return grp[batch == 1], grp[batch==2]
    