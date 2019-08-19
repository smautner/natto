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
            
            
    def process(self,markerfile,marker2=None, maxgenes=15, clust = 'gmm',sample=None,classpaths=None):
        
            markers, num = self.readmarkerfile(markerfile,maxgenes)
            if marker2:
                m,num2 = self.readmarkerfile(marker2,maxgenes)
                markers=markers.union(marker2)
            else:
                num2=num
            
            self.a = self.preprocess(self.a,markers)
            self.b = self.preprocess(self.b,markers)
            if clust == "gmm":
                x= self.a.X.toarray()
                y= self.b.X.toarray()
                self.mymap = umap.UMAP(n_components=6).fit(np.vstack((x,y)))
                x=self.mymap.transform(x)
                y=self.mymap.transform(y)
                clu1 = sim.predictgmm(num,x)
                clu2 = sim.predictgmm(num2,y)
                #return x,y,clu1, clu2
                
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
            
    def preprocess(self,adata,markers):
        '''we want to 0. basic filtering???  1.rows to 10k  2.select genes 3.normalize columns to 10xcells 4. log'''
        sc.pp.filter_cells(adata, min_genes=200)
        # rows to 10k
        sc.pp.normalize_total(adata,1e4)

        # select marker genes
        chose = [x in markers for x in adata.var['gene_ids'].keys() ]
        adata = adata[:,chose]

        # normalize column 
        adata.X = normalize(adata.X, axis=0, norm='l1')
        adata.X*=1e4
        # log
        sc.pp.log1p(adata)
        return adata


            
