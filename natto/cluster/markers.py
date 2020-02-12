import scanpy as sc
import numpy as np
from sklearn.preprocessing import normalize
load = lambda f: open(f,'r').readlines()
import natto.cluster.simple as sim
import umap
# so the idea is that we use seurat to get markers.

# then row norm 10k transcripts per cell
# then we select the columns with the markers 
# then we pp 10k and log them


class markers():
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
           
            
            
             
            
            
    def process2(self,maxgenes=15,corrcoef=True, dimensions=6, num=8):
        ####
        #  PREPROCESS, (normal single cell stuff)
        ########
        self.preprocess2(maxgenes)
            
            
        ####
        #  CLUSTER
        ########
        # TODO: predictgmm needs a num=-1 where we use the BIC to determine clustercount
        lena = self.a.shape[0]
        ax= self.a.X.toarray()
        bx= self.b.X.toarray()

        if corrcoef:
            corr = np.corrcoef(np.vstack((ax,bx))) 
            corr = np.nan_to_num(corr)
            ax,bx = corr[:lena], corr[lena:]
            
        self.mymap = umap.UMAP(n_components=dimensions).fit(np.vstack((ax,bx)))
        ax=self.mymap.transform(ax)
        bx=self.mymap.transform(bx)
        clu1 = sim.predictgmm_BIC(ax)
        clu2 = sim.predictgmm_BIC(bx)
        return ax,bx,clu1, clu2
            
    def preprocess2(self, maxgenes):
        self.cellfa,_=sc.pp.filter_cells(self.a, min_genes=200,inplace=False)
        self.cellfb,_=sc.pp.filter_cells(self.b, min_genes=200,inplace=False)
        self.a = self.a[self.cellfa,:]
        self.b = self.b[self.cellfb,:]
        #sc.pp.filter_genes(self.a, min_cells=3)
        #sc.pp.filter_genes(self.b, min_cells=3)
        #self.a.obs['n_counts'] = self.a.X.sum(axis=1).A1
        #self.b.obs['n_counts'] = self.b.X.sum(axis=1).A1
        sc.pp.normalize_total(self.a,1e4)
        sc.pp.normalize_total(self.b,1e4)
        sc.pp.log1p(self.a)
        sc.pp.log1p(self.b)
        
        sc.pp.highly_variable_genes(self.a,n_top_genes=maxgenes)
        sc.pp.highly_variable_genes(self.b,n_top_genes=maxgenes)
        
        
        genes= [f or g for f,g in zip(self.a.var.highly_variable, self.b.var.highly_variable)]
        self.a = self.a[:, genes]
        self.b = self.b[:, genes ]
        
 
        
        
        # they do this here: 
        # https://icb-scanpy-tutorials.readthedocs-hosted.com/en/latest/pbmc3k.html
        # sc.pp.regress_out(self.a, ['n_counts'])
        # sc.pp.regress_out(self.b, ['n_counts'])
        #sc.pp.scale(adata, max_value=10)

    
    
    

    '''
            elif clust == 'load':
                
                clu1 = self.csv_crap(classpaths[0])
                clu2 = self.csv_crap(classpaths[1])
                self.a.obs['class'] = clu1
                self.b.obs['class'] = clu2
                if sample: 
                    sc.pp.subsample(self.a, fraction=None, n_obs=sample, random_state=0, copy=False)
                    sc.pp.subsample(self.b, fraction=None, n_obs=sample, random_state=0, copy=False)
                clu1 = self.a.obs['class']
                clu2 = self.b.obs['class']
                
            else:
                clu1 = sim.predictlou(num,self.a.X.toarray(),{'n_neighbors':10})
                clu2 = sim.predictlou(num2,self.b.X.toarray(),{'n_neighbors':10})
            return self.a.X.toarray(), self.b.X.toarray(), clu1, clu2
        
    def csv_crap(self,f):
        lines =open(f,'r').readlines()
        return np.array([int(cls) for cls in lines[1:]])
            
    ''' 


            
