import scanpy as sc
import numpy as np
load = lambda f: [l for l in open(f,'r').read().split('\n') if len(l)>1]

#load = lambda f: open(f,'r').read()


def loadlabels(labels, ids):
    cellid_to_clusterid = {row.split(',')[0]:hash(row.split(',')[1]) for row in labels[1:]} #
    clusterid_to_nuclusterid = {item:clusterid for clusterid, item in enumerate(sorted(list(set(cellid_to_clusterid.values()))))}
    print (clusterid_to_nuclusterid)
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
        sc.pp.subsample(adata, fraction=.25, n_obs=None, random_state=0, copy=False)
    return adata

def load6k(cells: 'mito all seurat' ='mito', subsample=.25)-> 'anndata object':
    adata =  sc.read_10x_mtx(
    '../data/filtered_matrices_mex/hg19/',  
    var_names='gene_symbols', cache=True)

    adata.obs['labels']= loadlabels(load( "../data/pbmc.6k.labels"), load( "../data/filtered_matrices_mex/hg19/barcodes.tsv"))

    adata = filter(adata,cells)
    if subsample:
        sc.pp.subsample(adata, fraction=.25, n_obs=None, random_state=0, copy=False)
    return adata


def loadP7E(subsample=.25):
    adata = sc.read_10x_mtx( '../data/p7e',  var_names='gene_symbols', cache=True)
    if subsample:
        sc.pp.subsample(adata, fraction=.25, n_obs=None, random_state=0, copy=False)
    return adata

def loadP7D(subsample=.25):
    adata = sc.read_10x_mtx( '../data/p7d',  var_names='gene_symbols', cache=True)
    if subsample:
        sc.pp.subsample(adata, fraction=.25, n_obs=None, random_state=0, copy=False)
    return adata
