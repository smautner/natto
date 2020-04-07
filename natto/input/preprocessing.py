from lmz import * 
import ubergauss as ug
import scanpy as sc
import numpy as np
load = lambda f: open(f,'r').readlines()
import natto.old.simple as sim
import umap
from scipy.sparse import csr_matrix as csr
import sklearn 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
fun = lambda x,a,b,c: a+b/(1+x*c)
from scipy.stats import norm
from scipy import stats

class Data():
    """will have .a .b .d2 .dx"""
    def fit(self,adata, bdata,  maxgenes=100, corrcoef=True,mindisp=1.5,
                 dimensions=6,maxmean=3,scale=False, minmean=0.0125, debug_ftsel=False):

        self.a = adata
        self.b = bdata
        self.debug_ftsel = debug_ftsel
        # this will work on the count matrix:
        self.preprocess2( mindisp=mindisp,  maxmean=maxmean, minmean=minmean)
        #self.preprocess( maxgenes)
        if scale:
            self.scale()
            
        lena = self.a.shape[0]
        ax, bx = self._toarray()
        
        assert self.a.shape == ax.shape
        assert self.b.shape == bx.shape
        
        self.a, self.b  = ax,bx
        if corrcoef:
            corr = np.corrcoef(np.vstack((ax, bx)))
            corr = np.nan_to_num(corr)
            ax, bx = corr[:lena], corr[lena:]


        mymap = umap.UMAP(n_components=dimensions).fit(np.vstack((ax, bx)))
        self.dx = mymap.transform(ax), mymap.transform(bx)

        mymap = umap.UMAP(n_components=dimensions).fit(np.vstack((ax, bx)))
        self.d2 = mymap.transform(ax), mymap.transform(bx)
        return self
    


    def preprocess2(self, mindisp=1.5,maxmean = 3, minmean = None):
        self.basic_filter()
        self.normalize()
        a,b = self._toarray()
        
        #ag, _ = self.get_variable_genes(a, mindisp=mindisp, maxmean=maxmean, minmean=minmean)
        #bg,_  = self.get_variable_genes(b, mindisp=mindisp, maxmean = maxmean, minmean=minmean)
        #ag = self.get_var_genes_normtest(a,mindisp)
        #bg = self.get_var_genes_normtest(b,mindisp)
        ag = self.get_var_genes_simple(a, minmean,maxmean, cutoff= mindisp)
        bg = self.get_var_genes_simple(b,minmean,maxmean, cutoff = mindisp)
        genes = [a or b for a,b in zip(ag,bg)]
        self.a = self.a[:, genes].copy()
        self.b = self.b[:, genes].copy()
        
        
    def preprocess(self, maxgenes):
        self.basic_filter()
        self.normalize()

        # sophisticated feature selection
        Map(lambda x: sc.pp.highly_variable_genes(x, n_top_genes=maxgenes),[self.a,self.b])
        #genes = [f or g for f, g in zip(self.a.var.highly_variable, self.b.var.highly_variable)]
        genes = [f and g for f, g in zip(self.a.var.highly_variable, self.b.var.highly_variable)]
        self.a = self.a[:, genes].copy()
        self.b = self.b[:, genes].copy()

    
    def scale(self):
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
    
    def basic_filter(self):
        #self.a.X, self.b.X = self._toarray()
        # this weeds out obvious lemons (gens and cells)
        self.cellfa, gene_fa  = self._filter_cells_and_genes(self.a)
        self.cellfb, gene_fb  = self._filter_cells_and_genes(self.b)
        geneab = Map(lambda x, y: x or y, gene_fa, gene_fb)
        self.a = self.a[self.cellfa, geneab]
        self.b = self.b[self.cellfb, geneab]
        
    def normalize(self):
        sc.pp.normalize_total(self.a, 1e4)
        sc.pp.normalize_total(self.b, 1e4)
        sc.pp.log1p(self.a)
        sc.pp.log1p(self.b)
    
    
    
    def normtest(self, item):
        if len(item) < 8: 
            return 1 
        return stats.normaltest(item)[1]
    
    def get_var_genes_normtest(self,matrix, cutoff):
        result = []
        for i in range(matrix.shape[1]): # every feature
            values = [matrix[j,i] for j in range(matrix.shape[0]) if matrix[j,i]>0]
            result.append(self.normtest(values))
        result = np.array(result)
        ret =  result < cutoff 
        print('ok-genes:',sum(ret))
        return ret
        
            
    def get_var_genes_simple(self, matrix,minmean,maxmean , cutoff =.2):
        """not done yet """
        a=np.expm1(matrix)
        var     = np.var(a, axis=0)
        mean    = np.mean(a, axis=0)
        disp2= var /mean
        disp = np.log(disp2)
        mean = np.log1p(mean)
        #print (mean, disp2,maxmean,minmean)
        good = np.array( [not np.isnan(x) and me > minmean and me < maxmean for x,me in zip(disp2,mean)] )
        res= self.transform(mean[good].reshape(-1, 1),disp[good], ran = maxmean, minbin=1)
        
        mod= sklearn.linear_model.LinearRegression()
        mod.fit(*res)
        pre = mod.predict(mean[good].reshape(-1,1))
        good[good] = np.array([ (d-m)>cutoff for d,m in zip(disp[good],pre) ])
        if self.debug_ftsel: print(f"ft selected:{sum(good)}")
        return good 
        
        
        
        
    def get_variable_genes(self, matrix, minmean=0.0125, maxmean=3, mindisp=1.5):
        
        # get log-dispersion, log-mean, basic filtering (same as seurat)
        a=np.expm1(matrix)
        var     = np.var(a, axis=0)
        mean    = np.mean(a, axis=0)
        disp2= var /mean
        disp = np.log(disp2)
        mean = np.log1p(mean)
        good = np.array( [not np.isnan(x) and me > minmean and me < maxmean for x,me in zip(disp2,mean)] )
        
        # train lin model, get intercept and coef to transform 
        

        res= self.transform(mean[good].reshape(-1, 1),disp[good], ran = maxmean)
        
        '''
        mod= sklearn.linear_model.LinearRegression()
        mod.fit(*res)
        coef = mod.coef_[0]
        intercept =mod.intercept_
        # transform 
        disp-= intercept
        disp = np.array([ di/(me*mod.coef_[0]) for me,di in zip(mean,disp)])
        '''
        #disp = self.norm_dlin(disp,mean,res)
        #disp = self.norm_ceil(disp,mean,res)
        disp = self.norm_linear(disp,mean,res)
        
        disp[disp<-4] = -4
        disp[disp> 5] = 5
        
        # draw new values.... 
        if self.debug_ftsel:
            plt.scatter(mean[good],disp[good])
            plt.show()
        
        disp=disp.reshape(1,-1)[0]
        good[good] = np.array([ d>mindisp for d in disp[good] ])
        
        
        if self.debug_ftsel: print(f"ft selected:{sum(good)}")
            
        return good, disp # returning disp, for drawing purposes

    def transform(self,means,var, stepsize=.5, ran=3, minbin=0):
        x = np.arange(minbin*stepsize,ran,stepsize)
        items = [(m,v) for m,v in zip(means,var)]
        boxes = [ [i[1]  for i in items if r<i[0]<r+(stepsize) ]  for r in x  ]
        y = np.array([np.mean(st) for st in boxes])
        y_std = np.array([np.std(st) for st in boxes])
        x=x+(stepsize/2)

        # draw regression points 
        if self.debug_ftsel:plt.scatter(x,y)
        x = x.reshape(-1,1)
        return x,y,y_std
    
    def norm_dlin(self,disp,mean,res):
        ymod   = sklearn.linear_model.LinearRegression()
        stdmod = sklearn.linear_model.LinearRegression()
        ymod.fit(res[0],res[1])
        stdmod.fit(res[0],res[2])
        
        if self.debug_ftsel:
            plt.plot(res[0],ymod.predict(res[0]), color='green')
            plt.plot(res[0],stdmod.predict(res[0]), color='green')
            
        def nuvalue(x,y):
            m = ymod.predict(np.matrix([x]))
            s = stdmod.predict(np.matrix([x]))
            return norm.cdf(y,m,s)
            
        disp = np.array([ nuvalue(me,di) for me,di in zip(mean,disp)])
        return disp
    
    def norm_linear(self,disp,mean,res):
        mod= sklearn.linear_model.LinearRegression()
        mod.fit(*res)
        if self.debug_ftsel:plt.plot(res[0],mod.predict(res[0]), color='green')
            
        '''
        coef = mod.coef_[0]
        intercept = mod.intercept_+1
        disp-= intercept
        disp = np.array([ di/(1+(me*mod.coef_[0])) for me,di in zip(mean,disp)])
        '''
        #disp-=mod.predict(mean.reshape(-1,1))
        disp+= (-mod.intercept_)
        #disp/=mod.coef_[0]
        disp = np.array([ di/(me*mod.coef_[0]) for me,di in zip(mean,disp)])
        
        
        return disp
    
 

    def norm_ceil(self,disp,mean,res):
        x,y =res[0].reshape(1,-1)[0],res[1]
        plt.scatter(x,y)
        d1,d2 = curve_fit(fun,x,y)
        if self.debug_ftsel:plt.plot(x,[fun(xx,*d1) for xx in x], color='green')
        disp = np.array([ di-fun(me,*d1) for me,di in zip(mean,disp)])
        return disp

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
