from lmz import * 
import ubergauss as ug
import scanpy as sc
import numpy as np
load = lambda f: open(f,'r').readlines()
import natto.old.simple as sim
import umap
from scipy.sparse import csr_matrix as csr


class Data():
    """will have .a .b .umap_a .umap_b"""
    def fit(self,adata, bdata,  maxgenes=100, corrcoef=True,
                 dimensions=6, num=-1, scale=False):
        self.a = adata
        self.b = bdata

        # this will work on the count matrix:
        self.preprocess(maxgenes, scale)

        lena = self.a.shape[0]
        ax, bx = self.toarray()
        if corrcoef:
            corr = np.corrcoef(np.vstack((ax, bx)))
            corr = np.nan_to_num(corr)
            ax, bx = corr[:lena], corr[lena:]

        self.mymap = umap.UMAP(n_components=dimensions).fit(np.vstack((ax, bx)))
        self.umap_a = self.mymap.transform(ax)
        self.umap_b = self.mymap.transform(bx)

        return self

    def preprocess(self, maxgenes, scale=False):
        self.a.X, self.b.X = self.toarray()

        # this weeds out obvious lemons (gens and cells)
        cellfa, gene_fa  = self._filter_cells_and_genes(self.a)
        cellfb, gene_fb  = self._filter_cells_and_genes(self.b)
        geneab = Map(lambda x, y: x or y, gene_fa, gene_fb)
        self.a = self.a[cellfa, geneab].copy()
        self.b = self.b[cellfb, geneab].copy()

        # normalize:
        Map(lambda x: sc.pp.normalize_total(x, 1e4), [self.a, self.b])
        Map(lambda x: sc.pp.log1p(x), [self.a,self.b])

        # sophisticated feature selection
        Map(lambda x: sc.pp.highly_variable_genes(x, n_top_genes=maxgenes),[self.a,self.b])
        genes = [f or g for f, g in zip(self.a.var.highly_variable, self.b.var.highly_variable)]
        self.a = self.a[:, genes].copy()
        self.b = self.b[:, genes].copy()

        # they do this here:
        # https://icb-scanpy-tutorials.readthedocs-hosted.com/en/latest/pbmc3k.html
        # removed regout...
        if scale:
            sc.pp.scale(self.a, max_value=10)
            sc.pp.scale(self.b, max_value=10)


    ####
    # helper functions:
    ####
    def readmarkerfile(self, markerfile, maxgenes):
        """this reads a marker-gene file, it will extract and return: FEATURES, N_CLUSTERS"""
        markers_l = load(markerfile)
        markersets = [line.split(',') for line in markers_l[1:maxgenes]]
        numclusters = len(markersets[0])
        markers = {m.strip() for markerline in markersets for m in markerline}
        return markers, numclusters

    def _filter_cells_and_genes(self,ad):
        cellf, _ = sc.pp.filter_cells(ad, min_genes=200, inplace=False)
        genef, _ = sc.pp.filter_genes(ad, min_counts=6, inplace=False)
        return cellf, genef

    def _toarray(self):
        if isinstance(self.a.X, csr):
            ax = self.a.X.toarray()
            bx = self.b.X.toarray()
        else:
            ax = self.a.X
            bx = self.b.X
        return ax, bx



######
# old crap:
########
class markers_this_is_an_old_class():
    def __init__(self,adata,adata2):
        self.a = adata
        self.b = adata2


    def readmarkerfile(self,markerfile,maxgenes):
        markers_l = load(markerfile)
        markersets=[ line.split(',') for line in markers_l[1:maxgenes] ]
        numclusters = len(markersets[0])
        markers = {m.strip() for markerline in markersets for m in markerline}
            return markers, numclusters


            
    def transform(self):
        '''returns dim reduced data'''
        return self.mymap.transform(self.a.X.toarray()), self.mymap.transform(self.b.X.toarray())
    
    def process(self,markerfile,marker2=None, maxgenes=15, clust = 'gmm',sample=None,classpaths=None,corrcoef=True, dimensions=6):
        
            markers, num = self.readmarkerfile(markerfile,maxgenes)
            if marker2:
                m,num2 = self.readmarkerfile(marker2,maxgenes)
                markers=markers.union(marker2)
            else:
                num2=num
            
            self.a = self.preprocess(self.a,markers)
            self.b = self.preprocess(self.b,markers)
            lena = self.a.shape[0]

            ax= self.a.X.toarray()
            bx= self.b.X.toarray()

            if corrcoef:
                corr = np.corrcoef(np.vstack((ax,bx))) 
                corr = np.nan_to_num(corr)
                ax,bx = corr[:lena], corr[lena:]

            if clust == "gmm":
                self.mymap = umap.UMAP(n_components=dimensions).fit(np.vstack((ax,bx)))
                ax=self.mymap.transform(ax)
                bx=self.mymap.transform(bx)
                clu1 = sim.predictgmm(num,ax)
                clu2 = sim.predictgmm(num2,bx)
                return ax,bx,clu1, clu2
            
            
            
    def preprocess(self,adata,markers):
        '''we want to 0. basic filtering???  1.rows to 10k  2.select genes 3.normalize columns to 10xcells 4. log'''
        
        sc.pp.filter_cells(adata, min_genes=200)
        # rows to 10k
        sc.pp.normalize_total(adata,1e4)

        # select marker genes
        #chose = [x in markers for x in adata.var['gene_ids'].keys() ]
            
        chose = [x for x in adata.var['gene_ids'].keys() if x in markers]
        adata = adata[:,chose]

        # normalize column 
        #adata.X = normalize(adata.X, axis=0, norm='l1')
        #adata.X*=1e4
        # log
        sc.pp.log1p(adata)
        return adata
           
            
    def toarray(self):
        if isinstance(self.a.X, csr):
            ax= self.a.X.toarray()
            bx= self.b.X.toarray()
        else:
            ax= self.a.X
            bx= self.b.X
        return ax,bx
        
    def process2(self,maxgenes=15,corrcoef=True,
                 dimensions=6, num=-1,scale=False, regout = False):
        ####
        #  PREPROCESS, (normal single cell stuff)
        ########
        self.preprocess2(maxgenes, scale, regout)
            
            
        ####
        #  CLUSTER
        ########
        # TODO: predictgmm needs a num=-1 where we use the BIC to determine clustercount
        lena = self.a.shape[0]
        

        ax,bx = self.toarray()
            
            
        if corrcoef:
            corr = np.corrcoef(np.vstack((ax,bx))) 
            corr = np.nan_to_num(corr)
            ax,bx = corr[:lena], corr[lena:]
            
        self.mymap = umap.UMAP(n_components=dimensions).fit(np.vstack((ax,bx)))
        ax=self.mymap.transform(ax)
        bx=self.mymap.transform(bx)
    
        if num == -1:
            #clu1 = skc.DBSCAN().fit_predict(ax)
            #clu2 = skc.DBSCAN().fit_predict(bx)
            #clu1 = sim.predictgmm_mdk(ax)
            #clu2 = sim.predictgmm_mdk(bx)
            #clu1 = sim.fitbgmm(8,ax).predict(ax)
            #clu2 = sim.fitbgmm(8,bx).predict(bx)
            clu1 = ug.get_model(ax).predict(ax)
            clu2 = ug.get_model(bx).predict(bx)
            
            
            # TRY THIS ,,, ALSO TRY THE NORM BY COLUMN? 
            #clu1 = sim.predictlou(123,ax,{'n_neighbors':10})
            #clu2 = sim.predictlou(123,bx,{'n_neighbors':10})
        else:
            clu1 = sim.predictgmm(num,ax)
            clu2 = sim.predictgmm(num,bx)
        return ax,bx,clu1, clu2
    
    
    
    
    def preprocess2(self, maxgenes, scale=False, regout= False):
        
        self.a.X,self.b.X = self.toarray()
        
        self.cellfa,_=sc.pp.filter_cells(self.a, min_genes=200,inplace=False)
        self.cellfb,_=sc.pp.filter_cells(self.b, min_genes=200,inplace=False)
        geneb,_= sc.pp.filter_genes(self.b, min_counts=6,inplace=False)
        genea,_= sc.pp.filter_genes(self.a, min_counts=6,inplace=False)
        
        
        geneab = Map(lambda x,y: x or y , genea, geneb)
        self.a = self.a[self.cellfa,geneab].copy() 
        self.b = self.b[self.cellfb,geneab].copy()
        
        
        if regout:
            self.a.obs['n_counts'] = self.a.X.sum(axis=1)
            self.b.obs['n_counts'] = self.b.X.sum(axis=1)
            print("blabla",self.a.obs['n_counts'])
            
            
        sc.pp.normalize_total(self.a,1e4)
        sc.pp.normalize_total(self.b,1e4)
        sc.pp.log1p(self.a)
        sc.pp.log1p(self.b)
        sc.pp.highly_variable_genes(self.a,n_top_genes=maxgenes)
        sc.pp.highly_variable_genes(self.b,n_top_genes=maxgenes)
        
        
        genes= [f or g for f,g in zip(self.a.var.highly_variable, self.b.var.highly_variable)]
        self.a = self.a[:, genes].copy()
        self.b = self.b[:, genes].copy()
        
        # they do this here: 
        # https://icb-scanpy-tutorials.readthedocs-hosted.com/en/latest/pbmc3k.html
        if regout:
            print("regout",self.a.obs['n_counts'].shape ,self.a.shape)
            sc.pp.regress_out(self.a, ['n_counts'])
            sc.pp.regress_out(self.b, ['n_counts'])
        if scale:
            sc.pp.scale(self.a, max_value=10)
            sc.pp.scale(self.b, max_value=10)
