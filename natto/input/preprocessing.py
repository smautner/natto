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
from sklearn.preprocessing import PolynomialFeatures
class Data():
    """will have .a .b .d2 .dx"""
    def fit(self,adata, bdata,  
            maxgenes=750, 
            maxmean=3,
            mindisp=.25,
            minmean=0.0125, 
            corrcoef=True,
            dimensions=6,
            umap_n_neighbors = 15,
            pp='linear',
            scale=False,
            ft_combine = lambda x,y: x and y,
            debug_ftsel=False,
            make_even=False):

        self.a = adata
        self.b = bdata
        self.even = make_even  # will ssubsample to equal cells after cell outlayer rm
            
        self.debug_ftsel = debug_ftsel
        
        self.preprocess(pp=pp, 
                        scale=scale,
                        corrcoef=corrcoef,
                        mindisp=mindisp,
                        maxmean=maxmean,
                        ft_combine = ft_combine,
                        minmean=minmean,
                        maxgenes=maxgenes)
        
        #########
        # umap 
        ##########
        self.dx = self.umapify(dimensions, umap_n_neighbors)
        self.d2 = self.umapify(2, umap_n_neighbors)
        return self           
    
    def make_even(self):
        assert self.a.X.shape[1] == self.b.X.shape[1]
        if self.a.X.shape[0] > self.b.X.shape[0]:
            num=self.b.X.shape[0]
            target = self.a
        else:
            num= self.a.X.shape[0]
            target = self.b
        sc.pp.subsample(target,
                        fraction=None, 
                        n_obs=num, 
                        random_state=0, 
                        copy=False)
        
    def preprocess(self,pp='linear',
                   scale = False,
                   corrcoef = True,
                   ft_combine= lambda x,y: x or y,minbin=1, binsize=.25,
                   mindisp=.25, maxmean=3,minmean=0.015,maxgenes=750):
        
        if mindisp>0 and maxgenes>0:
            print ("processing data preprocess, needs explicit instructions on how to select features, defaulting to maxgenes")
        ###
        # prep
        ###
        self.norm_data()

        
        ######
        # SELECT GENES
        ########
        if pp == 'linear':
            ag,bg =self.preprocess_linear( mindisp=mindisp,
                                   maxmean=maxmean,
                                   minmean=minmean,
                                  maxgenes=maxgenes, 
                                    minbin=minbin,
                                    binsize=binsize)
        elif pp == 'bins':
            ag,bg = self.preprocess_bins( maxgenes)


        if pp == 'mergelinear':
            a,b= self._toarray()
            mat = np.vstack((a,b))
            genes = self.get_var_genes_linear( mat,minmean,maxmean,
                             cutoff = mindisp, Z= True, maxgenes=maxgenes, 
                             return_raw = False, minbin=minbin,binsize=binsize)
            print(f"genes: {sum(genes)}")
        else:
            #genes = [ft_combine(a,b) for a,b in zip(ag,bg)]
            genes = list(map(ft_combine,ag,bg))
            if self.debug_ftsel:
                print("number of features combined:", sum(genes))
            print(f"genes: {sum(genes)} fromA {sum(ag)} fromB {sum(bg)}")

        self.a = self.a[:, genes].copy()
        self.b = self.b[:, genes].copy()
        
        
        
        ######
        # finalizing 
        #####
        self.corrcoef(corrcoef,scale)
        
        
    


    
    def umapify(self, dimensions, n_neighbors):
        a,b= self._toarray()
        mymap = umap.UMAP(n_components=dimensions,n_neighbors=n_neighbors).fit(np.vstack((a, b)))
        return  mymap.transform(a), mymap.transform(b)




    def preprocess_linear(self,
                          mindisp=1.5,
                          maxmean = 3,
                          minmean = None,
                          maxgenes=None, minbin=1,binsize=.25):

        a,b = self._toarray()
        #ag, _ = self.get_variable_genes(a, mindisp=mindisp, maxmean=maxmean, minmean=minmean)
        #bg,_  = self.get_variable_genes(b, mindisp=mindisp, maxmean = maxmean, minmean=minmean)
        #ag = self.get_var_genes_normtest(a,mindisp)
        #bg = self.get_var_genes_normtest(b,mindisp)
        ag = self.get_var_genes_linear(a, minmean,maxmean,
                                       cutoff= mindisp,
                                       Z=True,
                                       minbin=minbin, binsize = binsize,
                                       maxgenes=maxgenes)
        self.prevres =  ag
        bg = self.get_var_genes_linear(b, minmean,maxmean,
                                       cutoff = mindisp, 
                                       maxgenes=maxgenes,
                                       minbin=minbin, binsize = binsize,
                                       Z=True)
        
        return ag,bg

        
        
    def preprocess_bins(self, maxgenes):
        Map(lambda x: sc.pp.highly_variable_genes(x, n_top_genes=maxgenes),[self.a,self.b])
        #genes = [f or g for f, g in zip(self.a.var.highly_variable, self.b.var.highly_variable)]
        return self.a.var.highly_variable, self.b.var.highly_variable

    


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

    def _filter_cells_and_genes(self,ad, min_genes=200, min_counts=6):
        cellf, _ = sc.pp.filter_cells(ad, min_genes=min_genes, inplace=False)
        genef, _ = sc.pp.filter_genes(ad, min_counts=min_counts, inplace=False)
        return cellf, genef


    def _toarray(self): 
        return self.__toarray(self.a), self.__toarray(self.b)

    def __toarray(self,thing):       
        if isinstance(thing, np.ndarray):
            return thing
        elif isinstance(thing, csr):
            return thing.toarray()
        elif isinstance(thing.X, np.ndarray):
            return thing.X
        elif isinstance(thing.X, csr):
            return  thing.X.toarray()
        print("type problem in to array ")
    
    def basic_filter(self, min_counts=6, min_genes=200):
        #self.a.X, self.b.X = self._toarray()
        # this weeds out obvious lemons (gens and cells)
        self.cellfa, gene_fa  = self._filter_cells_and_genes(self.a, min_genes, min_counts)
        self.cellfb, gene_fb  = self._filter_cells_and_genes(self.b, min_genes, min_counts)
        geneab = Map(lambda x, y: x or y, gene_fa, gene_fb)
        self.a = self.a[self.cellfa, geneab]
        self.b = self.b[self.cellfb, geneab]
        
    def normalize(self):
        sc.pp.normalize_total(self.a, 1e4)
        sc.pp.normalize_total(self.b, 1e4)
        sc.pp.log1p(self.a)
        sc.pp.log1p(self.b)
    
 
    def norm_data(self):
        self.basic_filter()
        self.normalize()   
        if self.even: 
            self.make_even()
    

    ####
    # ft select 
    ###
    def transform(self,means,var, stepsize=.5, ran=3, minbin=0):
        x = np.arange(minbin*stepsize,ran,stepsize)
        items = [(m,v) for m,v in zip(means,var)]
        boxes = [ [i[1]  for i in items if r<i[0]<r+(stepsize) ]  for r in x  ]
        y = np.array([np.mean(st) for st in boxes])
        y_std = np.array([np.std(st) for st in boxes])
        x=x+(stepsize/2)
        # draw regression points 
        if self.debug_ftsel:plt.scatter(x,y,label='Mean of bins')


        nonan=np.isfinite(y)
        x= x[nonan]
        y=y[nonan]
        y_std = y_std[nonan]

        x = x.reshape(-1,1)
        return x,y,y_std       
    
    
    def generalize_quadradic(self,y,x):
        
        poly_reg=PolynomialFeatures(degree=2)
        x=poly_reg.fit_transform(x.reshape(-1,1))
        mod= sklearn.linear_model.LinearRegression()
        mod.fit(x,y)
        return mod.predict(x)
    
    def get_expected_values(self,x,y,x_all):
        #mod= sklearn.linear_model.LinearRegression()
        #mod= sklearn.linear_model.RANSACRegressor()
        mod= sklearn.linear_model.HuberRegressor()
        #mod.fit(x_all[x_all >= x[0]].reshape(-1,1),y_all[x_all >= x[0]]) # ...
        mod.fit(x,y)
        res = mod.predict(x_all.reshape(-1,1))
        #res[x_all < x[0]] = y[0] # produces a harsh step...
        res[x_all <  x[0] ] = mod.predict([x[0]])
        return res

    def get_var_genes_linear(self, matrix,minmean,maxmean,
                             cutoff =.2, Z= True, maxgenes=None, 
                             return_raw = False, minbin=1,binsize=.25):
        
        if maxgenes and not Z: 
            print ("maxgenes without Z transform is meaningless")
            
        """not done yet """
        a=np.expm1(matrix)
        var     = np.var(a, axis=0)
        mean    = np.mean(a, axis=0)
        disp2= var/mean
        Y = np.log(disp2)
        X = np.log1p(mean)

        #print (mean, disp2,maxmean,minmean)
        good = np.array( [not np.isnan(y) and me > minmean and me < maxmean for y,me in zip(disp2,X)] )
        
        if self.debug_ftsel: 
            ax=plt.subplot(121)
            plt.scatter(X[good], Y[good],alpha=.2, s=3, label='all')
            
        x_bin,y_bin,ystd_bin = self.transform(X[good].reshape(-1, 1),Y[good],stepsize=binsize, ran = maxmean, minbin=minbin)
        

        pre = self.get_expected_values(x_bin,y_bin,X[good])
        ###
        # make it quadratic
        ####
        #pre = self.generalize_quadradic(pre,X[good])
        
        if Z:
            std = self.get_expected_values(x_bin,ystd_bin,X[good])
            Y[good]-= pre 
            Y[good]/=std
            if not maxgenes:
                accept = [d > cutoff for d in Y[good]]                
                if return_raw:
                    bad=np.logical_not(good)
                    
                    pre_bad = self.get_expected_values(x_bin,y_bin,X[bad])
                    std_bad = self.get_expected_values(x_bin,ystd_bin,X[bad])
                    Y[bad] -= pre_bad
                    Y[bad] /= std_bad
                    return Y
            else: 
                srt = np.argsort(Y[good])
                accept = np.full(Y[good].shape, False)
                accept[ srt[-maxgenes:] ] = True

            
        else:
            accept = [ (d-m)>cutoff for d,m in zip(Y[good],pre) ]
        
        
        if self.debug_ftsel:
            srt= np.argsort(X[good])
            plt.plot(X[good][srt], pre[srt],color='k', label='regression')
            plt.scatter(x_bin, ystd_bin, alpha= .4, label='Std')
            plt.legens()
            ax=plt.subplot(122)
            plt.scatter(X[good], Y[good],alpha=.2, s=3, label = 'all')

            g=X[good]
            d=Y[good]
            plt.scatter(g[accept], d[accept],alpha=.3, s=3, color='r', label='selected')

            if  "prevres" in self.__dict__:
                argh = good.copy()
                argh[good] = np.array(accept)
                agree = argh & self.prevres
                plt.scatter(X[agree], Y[agree],alpha=.8, s=3, color='k', label='overlap')

            plt.legend()
            plt.show()

            print(f"ft selected:{sum(accept)}")
        
        good[good] = np.array(accept)
        
        return good 
    
    def __init__(self):
        self.debug_ftsel = False

    def scale(self):
        sc.pp.scale(self.a, max_value=10)
        sc.pp.scale(self.b, max_value=10)

    def corrcoef(self, corrcoef, scale):
        ########
        # corrcoef
        ########
        if scale:
            self.scale()      
        if corrcoef:
            ax, bx = self._toarray()
            lena = self.a.shape[0]
            corr = np.corrcoef(np.vstack((ax, bx)))
            corr = np.nan_to_num(corr)
            self.a, self.b = corr[:lena], corr[lena:]
        return    
"""
         
    def preprocess_simple(self, maxgenes=700):

        a,b = self._toarray()
        #ag, _ = self.get_variable_genes(a, mindisp=mindisp, maxmean=maxmean, minmean=minmean)
        #bg,_  = self.get_variable_genes(b, mindisp=mindisp, maxmean = maxmean, minmean=minmean)
        #ag = self.get_var_genes_normtest(a,mindisp)
        #bg = self.get_var_genes_normtest(b,mindisp)
        A=np.expm1(a)
        B=np.expm1(b)
        def select(X):
            var = X.var(axis=0)
            srt = np.argsort(var)
            accept = srt>(srt.shape[0]-maxgenes)
            if self.debug_ftsel: 
                print(f"ft selected:{sum(accept)}")
            return accept
            
        return select(A),select(B)
        
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

"""
