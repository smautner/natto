
from lmz import *
import ubergauss as ug
import scanpy as sc
import numpy as np
load = lambda f: open(f ,'r').readlines()
import umap
from scipy.sparse import csr_matrix as csr
import sklearn
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
fun = lambda x ,a ,b ,c: a+ b / (1 + x * c)
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import natto.input.hungarian as h


class Data():
    """will have .a .b .d2 .dx"""

    def fit(self, adata_list,
            maxgenes=100,
            maxmean=4,
            mindisp=False,
            minmean=0.015,
            dimensions=10,
            umap_n_neighbors=10,
            pca=20,
            ft_combine=lambda x, y: x or y,
            debug_ftsel=True,
            mitochondria=False,
            titles="abcdefghijklmnop",
            scale=True,
            quiet=False,
            make_even=True):

        # assert adata.var["gene_ids"].index ==  bdata.var["gene_ids"].index
        self.mitochondria = mitochondria
        self.data = adata_list
        self.titles = titles
        self.even = make_even  # will ssubsample to equal cells after cell outlayer rm
        self.debug_ftsel = debug_ftsel
        self.quiet = quiet

        self.preprocess( mindisp=mindisp,
                        maxmean=maxmean,
                        ft_combine=ft_combine,
                        minmean=minmean,
                        maxgenes=maxgenes)
        #########
        # umap
        ##########
        self.dimension_reduction(pca, dimensions, umap_n_neighbors, scale=scale)
        self.sort_cells()
        return self

    
    def sort_cells(self):
        
        #assert False, "not implemented"

        for i in range(len(self.data)-1):
            hung, _ = h.hungarian(self.dx[i],self.dx[i+1])
            self.data[i+1].X = self.data[i+1].X[hung[1]]
            self.dx[i+1] = self.dx[i+1][hung[1]]
            self.d2[i+1] = self.d2[i+1][hung[1]]
            #self.pca[i+1] = self.pca[i+1][hung[1]]


    def dimension_reduction(self, pca, dimensions, umap_n_neighbors, scale= True):
        self.mk_pca(pca, scale= scale)
        self.dx = self.umapify(dimensions, umap_n_neighbors)
        self.d2 = self.umapify(2, umap_n_neighbors)



    def printcounts(self, where):
        counts  = [  e.X.shape[0] for e in self.data  ]
        print('printcounts:', where, counts)

    def make_even(self):

        # assert all equal 
        size = self.data[0].X.shape[1]
        assert all ([size == other.X.shape[1] for other in self.data])

        # find smallest
        counts  = [  e.X.shape[0] for e in self.data  ]
        smallest = min(counts)

        

        for a in self.data:
            if a.X.shape[0] > smallest:
                sc.pp.subsample(a,
                        fraction=None,
                        n_obs=smallest,
                        random_state=0,
                        copy=False)



    def preprocess(self, pp='linear',
                   ft_combine=lambda x, y: x or y,
                   minbin=1,
                   binsize=.25,
                   mindisp=.25,
                   maxmean=3,
                   minmean=0.015,
                   maxgenes=750):

        if mindisp > 0 and maxgenes > 0:
            print( "processing data preprocess, needs explicit instructions on how to select features, defaulting to maxgenes")
        ###
        # prep
        ###
        self.norm_data()

        ######
        # SELECT GENES
        ########
        genelists = self.preprocess_linear(mindisp=mindisp,
                                        maxmean=maxmean,
                                        minmean=minmean,
                                        maxgenes=maxgenes,
                                        minbin=minbin,
                                        binsize=binsize)

        # genes = [ft_combine(a,b) for a,b in zip(ag,bg)]
        #genes = list(map(ft_combine, ag, bg))
        #if self.debug_ftsel:
        #    print("number of features combined:", sum(genes))
        #if not self.quiet: print(f"genes: {sum(genes)} fromA {sum(ag)} fromB {sum(bg)}")
        genes  = np.any(np.array(genelists), axis  = 0)  ##???? lets see if this works
        self.data = [ d[:,genes].copy() for d in self.data  ]

    def scale(self):
        [sc.pp.scale(adat, max_value =10) for adat in self.data]

    def mk_pca(self, PCA, scale = True):


        if scale:
            self.scale()

        read_mat = self._toarray()

        if not PCA: 
            self.pca = read_mat, PCA
            return read_mat

        # we do PCA
        if scale == False: # if scale is false we scale all together to make pca vaible
            scaler= StandardScaler()
            scaled = [ scaler.fit_transform(m) for m in read_mat]
        else:
            scaled= read_mat


        pca = sklearn.decomposition.PCA(n_components=PCA)
        pca.fit(np.vstack(scaled))
        blocks = [ pca.transform(e) for e in scaled  ]


        self.pca = blocks, PCA
        return blocks

    def umapify(self, dimensions, n_neighbors):
        stuff, pcadim = self.pca
        if 0 < pcadim <= dimensions:
            return stuff

        mymap = umap.UMAP(n_components=dimensions,
                          n_neighbors=n_neighbors,
                          random_state=1337).fit(
            np.vstack(stuff))

        return [ mymap.transform(s) for s in stuff]



    def preprocess_linear(self,
                          mindisp=1.5,
                          maxmean=3,
                          minmean=None,
                          maxgenes=None, minbin=1, binsize=.25):

        blocks = self._toarray()
        return [self.get_var_genes_linear(b, minmean, maxmean,
                                       cutoff=mindisp,
                                       maxgenes=maxgenes,
                                       minbin=minbin,
                                       binsize=binsize, title=self.titles[i],
                                       Z=True)  for i,b in enumerate(blocks)]






    def _toarray(self):
        return [self.__toarray(x) for x in self.data]

    def __toarray(self, thing):
        if isinstance(thing, np.ndarray):
            return thing
        elif isinstance(thing, csr):
            return thing.toarray()
        elif isinstance(thing.X, np.ndarray):
            return thing.X
        elif isinstance(thing.X, csr):
            return thing.X.toarray()
        print("type problem in to array ")



    def basic_filter(self, min_counts=3, min_genes=200):

        # filter cells
        [sc.pp.filter_cells(d, min_genes=min_genes, inplace=True) for d in self.data]

        # filter genes
        genef  = [ sc.pp.filter_genes(d, min_counts=min_counts, inplace=False)[0] for d in self.data]
        geneab = np.any(np.array(genef),axis = 0)
        for i,d in enumerate(self.data):
            self.data[i] = d[:,geneab]





    def normalize(self):

        if self.mitochondria:
            for d in self.data:
                d.var['mt'] = d.var_names.str.startswith(self.mitochondria)
                sc.pp.calculate_qc_metrics(d, qc_vars=['mt'], percent_top=None, inplace=True)

            for i,d in enumerate(self.data):
                self.data[i] = d[d.obs.pct_counts_mt < 5, :]

        [sc.pp.normalize_total(d, 1e4) for d  in self.data]
        [sc.pp.log1p(d) for d in self.data]

    def norm_data(self):
        self.basic_filter()
        self.normalize()
        if self.even:
            self.make_even()

    ####
    # ft select
    ###
    def transform(self, means, var, stepsize=.5, ran=3, minbin=0, bin_avg='mean'):
        x = np.arange(minbin * stepsize, ran, stepsize)
        items = [(m, v) for m, v in zip(means, var)]
        boxes = [[i[1] for i in items if r < i[0] < r + (stepsize)] for r in x]
        if bin_avg == 'mean':
            y = np.array([np.mean(st) for st in boxes])
        elif bin_avg == 'median':
            y = np.array([np.median(st) for st in boxes])
        else:
            assert False
        y_std = np.array([np.std(st) for st in boxes])
        x = x + (stepsize / 2)
        # draw regression points
        if self.debug_ftsel: plt.scatter(x, y, label='Mean of bins', color='k')

        nonan = np.isfinite(y)
        x = x[nonan]
        y = y[nonan]
        y_std = y_std[nonan]

        x = x.reshape(-1, 1)
        return x, y, y_std

    def generalize_quadradic(self, y, x):

        poly_reg = PolynomialFeatures(degree=2)
        x = poly_reg.fit_transform(x.reshape(-1, 1))
        mod = sklearn.linear_model.LinearRegression()
        mod.fit(x, y)
        return mod.predict(x)

    def get_expected_values(self, x, y, x_all, leftmost_values='const_max'):
        # mod= sklearn.linear_model.LinearRegression()
        # mod= sklearn.linear_model.RANSACRegressor()
        mod = sklearn.linear_model.HuberRegressor()
        # mod.fit(x_all[x_all >= x[0]].reshape(-1,1),y_all[x_all >= x[0]]) # ...
        mod.fit(x, y)
        res = mod.predict(x_all.reshape(-1, 1))

        firstbin = y[0]
        firstbin_esti = mod.predict([x[0]])

        if leftmost_values == 'const_max':
            res[x_all < x[0]] = max(firstbin, firstbin_esti)
        elif leftmost_values == 'nointerference':
            pass
        elif leftmost_values == 'const_firstbin':
            res[x_all < x[0]] = firstbin
        elif leftmost_values == 'const_firstbin_esti':
            res[x_all < x[0]] = firstbin_esti
        else:
            assert False

        return res

    def get_var_genes_linear(self, matrix, minmean, maxmean,
                             cutoff=.2, Z=True, maxgenes=None,
                             return_raw=False, minbin=1, binsize=.25, title='None'):

        if maxgenes and not Z:
            print("maxgenes without Z transform is meaningless")

        a = np.expm1(matrix)
        var = np.var(a, axis=0)
        mean = np.mean(a, axis=0)
        disp = var / mean

        Y = np.log(disp)
        X = np.log1p(mean)

        mask = np.array([not np.isnan(y) and me > minmean and me < maxmean for y, me in zip(disp, X)])

        if self.debug_ftsel:
            plt.figure(figsize=(11, 4))
            plt.suptitle(f"gene selection: {title}", size=20, y=1.07)
            ax = plt.subplot(121)
            plt.scatter(X[mask], Y[mask], alpha=.2, s=3, label='all genes')

        x_bin, y_bin, ystd_bin = self.transform(X[mask].reshape(-1, 1),
                                                Y[mask],
                                                stepsize=binsize,
                                                ran=maxmean,
                                                minbin=minbin,
                                                bin_avg='median')  # median or mean

        y_predicted = self.get_expected_values(x_bin, y_bin, X[mask])
        ###
        # make it quadratic
        ####
        # pre = self.generalize_quadradic(pre,X[good])

        if Z:
            std_predicted = self.get_expected_values(x_bin, ystd_bin, X[mask])
            Y[mask] -= y_predicted
            Y[mask] /= std_predicted
            if not maxgenes:
                accept = [d > cutoff for d in Y[mask]]
                if return_raw:
                    # Y[mask] is already corrected, now we correct the complement and return raw
                    bad = np.logical_not(mask)
                    pre_bad = self.get_expected_values(x_bin, y_bin, X[bad])
                    std_bad = self.get_expected_values(x_bin, ystd_bin, X[bad])
                    Y[bad] -= pre_bad
                    Y[bad] /= std_bad
                    return Y
            else:
                srt = np.argsort(Y[mask])
                accept = np.full(Y[mask].shape, False)
                accept[srt[-maxgenes:]] = True


        else:
            accept = [(d - m) > cutoff for d, m in zip(Y[mask], y_predicted)]

        if self.debug_ftsel:
            srt = np.argsort(X[mask])
            plt.plot(X[mask][srt], y_predicted[srt], color='k', label='regression')
            plt.plot(X[mask][srt], std_predicted[srt], color='g', label='regression of std')
            plt.scatter(x_bin, ystd_bin, alpha=.4, label='Std of bins', color='g')
            plt.legend(bbox_to_anchor=(.6, -.2))
            plt.title("dispersion of genes")
            plt.xlabel('log mean expression')
            plt.ylabel('dispursion')

            ax = plt.subplot(122)
            plt.scatter(X[mask], Y[mask], alpha=.2, s=3, label='all genes')

            g = X[mask]
            d = Y[mask]
            plt.scatter(g[accept], d[accept], alpha=.3, s=3, color='r', label='selected genes')

            if "prevres" in self.__dict__:
                argh = mask.copy()
                argh[mask] = np.array(accept)
                agree = argh & self.prevres
                plt.scatter(X[agree], Y[agree], alpha=.8, s=3, color='k', label='selected genes overlap')

            plt.legend(bbox_to_anchor=(.6, -.2))
            plt.title("normalized dispersion of genes")
            plt.xlabel('log mean expression')
            plt.ylabel('dispursion')
            plt.show()

            print(f"ft selected:{sum(accept)}")

        mask[mask] = np.array(accept)

        return mask

    def __init__(self):
        self.debug_ftsel = False



