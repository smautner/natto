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
            maxmean=7,
            mindisp=1.4,
            minmean=0.0125, 
            corrcoef=True,
            dimensions=6,
            umap_n_neighbors = 15,
            pp='linear',
            scale=False,
            umap_pca = False, 
            ft_combine = lambda x,y: x or y,
            debug_ftsel=False,
            mitochondria = False, 
            titles = ("no title set in data constructure","<-"), 
            make_even=False):
        self.mitochondria = mitochondria
        self.a = adata
        self.b = bdata
        self.titles = titles
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
        self.dx = self.umapify(dimensions, umap_n_neighbors, PCA = umap_pca)
        self.d2 = self.umapify(2, umap_n_neighbors, PCA=umap_pca)
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
            print(f"genes: {sum(genes)} / {len(genes)}")
        else:
            #genes = [ft_combine(a,b) for a,b in zip(ag,bg)]
            genes = list(map(ft_combine,ag,bg))
            if self.debug_ftsel:
                print("number of features combined:", sum(genes))
            print(f"genes: {sum(genes)} fromA {sum(ag)} fromB {sum(bg)}")


        
        self.a = self.a[:, genes].copy()
        self.b = self.b[:, genes].copy()
        if self.mitochondria:
            pass
            '''
            self.basic_filter() 
            self.a.X, self.b.X = self._toarray()
            print(np.sum(self.a.X.sum(axis=0) == 0))  # this is true, HOW? 
            print(np.sum(self.a.X.sum(axis=1) == 0))
            print(np.sum(self.b.X.sum(axis=0) == 0)) 
            print(np.sum(self.b.X.sum(axis=1) == 0))
            sc.pp.regress_out(self.a, ['total_counts', 'pct_counts_mt'])
            sc.pp.regress_out(self.b, ['total_counts', 'pct_counts_mt'])
            '''

        ######
        # finalizing 
        #####
        self.corrcoef(corrcoef,scale)
        
        
    


    
    def umapify(self, dimensions, n_neighbors, PCA = 0):
        
        a,b= self._toarray()

        if PCA:
            ab= np.vstack((a, b))
            pca = sklearn.decomposition.PCA(n_components=30)
            ab = pca.fit_transform(ab)

            if False:# kneedler
                import kneed 
                var = pca.explained_variance_ratio_
                kneeder = kneed.KneeLocator(Range(30), var,S=5.0,  curve='convex', direction = 'decreasing')
                #kneeder.plot_knee()
                v = kneeder.knee
                print(f'selecting {v} pca vectors')
                #a = ab[:len(a),:v+1]
                #b = ab[len(a):,:v+1]
            a = ab[:len(a)]
            b = ab[len(a):]


        mymap = umap.UMAP(n_components=dimensions,n_neighbors=n_neighbors).fit(
                np.vstack((a, b)))
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

        if self.mitochondria:
            # BLABLA DO STUFF 
            mitochondria= ad.var['gene_ids'].index.str.match(f'^{self.mitochondria}.*')
            rowcnt = ad.X.sum(axis =1) 
            row_mitocnt  = ad.X[:,mitochondria].sum(axis=1)
        
            mitoarray = (row_mitocnt/rowcnt) < .05
            print(f"filtering mito {sum( (row_mitocnt/rowcnt) > .05  )}  genes{sum(mitochondria)} ")
            cellf = np.logical_and(cellf,mitoarray.getA1())
            
            

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
    
    def basic_filter(self, min_counts=3, min_genes=200):
        #self.a.X, self.b.X = self._toarray()
        # this weeds out obvious lemons (gens and cells)
        self.cellfa, gene_fa  = self._filter_cells_and_genes(self.a, min_genes, min_counts)
        self.cellfb, gene_fb  = self._filter_cells_and_genes(self.b, min_genes, min_counts)
     
        
        geneab = Map(lambda x, y: x or y, gene_fa, gene_fb)
        self.a = self.a[self.cellfa, geneab]
        self.b = self.b[self.cellfb, geneab]
        
    def normalize(self):

        if self.mitochondria:
            self.a.var['mt'] = self.a.var_names.str.startswith(self.mitochondria)  
            sc.pp.calculate_qc_metrics(self.a, qc_vars=['mt'], percent_top=None, inplace=True)

            self.b.var['mt'] = self.b.var_names.str.startswith(self.mitochondria)  
            sc.pp.calculate_qc_metrics(self.b, qc_vars=['mt'], percent_top=None, inplace=True)
            #print (f"doing mito: {sum(self.a.obs.pct_counts_mt < 5)}")
            #print (list(self.a.obs.pct_counts_mt))
            self.a = self.a[self.a.obs.pct_counts_mt < 5, :]
            self.b = self.b[self.b.obs.pct_counts_mt < 5, :]

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
    def transform(self,means,var, stepsize=.5, ran=3, minbin=0, bin_avg = 'mean'):
        x = np.arange(minbin*stepsize,ran,stepsize)
        items = [(m,v) for m,v in zip(means,var)]
        boxes = [ [i[1]  for i in items if r<i[0]<r+(stepsize) ]  for r in x  ]
        if bin_avg == 'mean':
            y = np.array([np.mean(st) for st in boxes])
        elif bin_avg == 'median':
            y = np.array([np.median(st) for st in boxes])
        else:
            assert False
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
    
    def get_expected_values(self,x,y,x_all, smooth_leftmost_values=False):
        #mod= sklearn.linear_model.LinearRegression()
        #mod= sklearn.linear_model.RANSACRegressor()
        mod= sklearn.linear_model.HuberRegressor()
        #mod.fit(x_all[x_all >= x[0]].reshape(-1,1),y_all[x_all >= x[0]]) # ...
        mod.fit(x,y)
        res = mod.predict(x_all.reshape(-1,1))
        if not smooth_leftmost_values:
            res[x_all < x[0]] = y[0] # produces a harsh step...
        else:
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
            plt.figure(figsize=(10,4)) 
            ax=plt.subplot(121)
            plt.scatter(X[good], Y[good],alpha=.2, s=3, label='all')
            
        x_bin,y_bin,ystd_bin = self.transform(X[good].reshape(-1, 1),
                        Y[good],
                        stepsize=binsize, 
                        ran = maxmean,
                        minbin=minbin,
                        bin_avg = 'median') # median or mean
        

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
            plt.legend()
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
