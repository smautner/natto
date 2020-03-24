load = lambda f: open(f,'r').readlines()
#  we load the data from clustermap
from collections import defaultdict

class clustermapdata():
    
    
    
    
    def readlabels(self,dataset):
        file = f"../dataset_rdses/{dataset}.clusters.csv"
        f = load(file)[1:] # ignore header
        return [int(line[line.find(',')+2:-2]) for line in f]


    def readmarkerfile(self,dataset,maxgenes):
        file = f"../dataset_rdses/{dataset}.markers.csv"
        markers_l = load(file)
        d = defaultdict(list)
        markersets=[ line.split(',') for line in markers_l[1:] ]
        for line in markersets:
            d[int(line[-2][1:-1])].append(line[0][1:-1])
        print(d.keys())
        numclusters = len(markersets[0])
        markers = { m  for markers in d.values()  for m in markers[:maxgenes] }
        return markers, len(d)


    
    
    # step1 load data... 
    #       load usual stuff
    #       load class labels 
    #       subsample based on this 
    
    def __init__(self,adata,adata2): # we should make sure that we are not subsampling
        
        
        self.a = adata
        self.b = adata2
        # this is how subsampling works..
        #sc.pp.subsample(adata, fraction=subsample, n_obs=None, random_state=0, copy=False)
        
        
    def asd(self):    
        # step2 return clusters and data that selects the marker genes
        #       load clusters.csv
        pass
 
""" this is from the marker loader... we build something similar here

    def process(self,markerfile,marker2=None, maxgenes=15, clust = 'gmm',sample=None,classpaths=None, dimensions=6):
        
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
                self.mymap = umap.UMAP(n_components=dimensions).fit(np.vstack((x,y)))
                x=self.mymap.transform(x)
                y=self.mymap.transform(y)
                clu1 = sim.predictgmm(num,x)
                clu2 = sim.predictgmm(num2,y)
                # return x,y,clu1, clu2
                
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


"""
