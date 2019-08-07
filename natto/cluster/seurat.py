import numpy as np
import basics as b
import scanpy as sc
from sklearn import decomposition as skdecomp
import umap



class reseurat():
    def __init__(self):
        self.values={}
    
    def wog(self,name,value):
        if name in self.values:
            return self.values[name]
        else:
            self.values[name] = value
            return value
        
    def soy(self,adata,makeunique=True):
        # bla
        if makeunique:
            adata.var_names_make_unique()
        sc.pp.filter_cells(adata, min_genes=200)
        
        #! genes: cellcount >= 3   
        genes,_ = sc.pp.filter_genes(adata, min_cells=3,inplace=False)
        select = self.wog("genemincell", genes)
        adata = adata[:,select]
        
        
        # mito and counts, normalize 
        mito_genes = adata.var_names.str.startswith('MT-')
        adata.obs['percent_mito'] = np.sum(
            adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
        adata.obs['n_counts'] = adata.X.sum(axis=1).A1
        adata = adata[adata.obs['n_genes'] < 2500, :]
        adata = adata[adata.obs['percent_mito'] < 0.05, :]
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
        sc.pp.log1p(adata) 
        
                          
        #! variable genes
        vari = sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, inplace=False)['highly_variable']
        vari = self.wog("highly variable",vari)
        adata = adata[:, vari ]
        
        
        #!! this is the highly problematic regress out step
        if 'regout' in self.values:
            dispersions  = sc.pp.highly_variable_genes(adata, min_disp=0.1, inplace=False)['highly_variable']
            adata2 = adata[:,dispersions] 
            sc.pp.regress_out(adata2, ['n_counts', 'percent_mito'])
            update_anndata(adata,adata2,dispersions)
        
        else:
            sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])
            self.values['regout']=True
        
        # this is ok ;; somehow nans are added here
        sc.pp.scale(adata, max_value=10)
        
        #! PCA 
        if 'pca' not in self.values:
            pca = skdecomp.PCA(n_components=40)
            pca.fit(adata.X)
            self.values['pca'] = pca

        adata.X=np.nan_to_num(adata.X)

        adata.obsm['X_pca'] = self.values['pca'].transform(adata.X)
        sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_pca')
        sc.tl.louvain(adata)
        adata.obs['louvain'] = b.lmap(int,adata.obs['louvain'])
        
        
        reducer = umap.UMAP()
        reducer.fit(adata.obsm['X_pca'])
        reducer = self.wog('reducer', reducer)
        return adata, reducer 
    
    
    

def update_anndata_assuming_csr(adata,adata2,where): 
    '''
    adata: big
    adata2:smaller
    where: bool list ; sum(true)= len(smaller)
    '''
    where = [i for i,b in enumerate(where) if b]
    for i in range(adata.X.shape[0]):
        a,b = adata2.X.indptr[i], adata2.X.indptr[i+1]
        ind = adata2.X.indices[a:b]
        dat = adata2.X.data[a:b]
        for j,d in zip(ind,dat):
            adata.X[i,where[j]] = d
            
def update_anndata(adata,adata2,where):
    # assuming adata2 is quite large and not csr
    '''
    adata: big
    adata2:smaller
    where: bool list ; sum(true)= len(smaller)
    '''
    adata.X= adata.X.toarray()
    where = [i for i,b in enumerate(where) if b]
    for i,j in zip(range(adata2.shape[1]), where):
        adata.X[:,j]=adata2.X[:,i]
        

