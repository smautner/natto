from natto.process import preprocess
import scanpy as sc
from natto.process import dimensions
from natto.process import util as u


class Data():
    def fit(self,adatas,
            selector='natto',
            selectgenes=800,

            meanexp = (0.015,4),
            bins = (.25,1),
            visual_ftsel=True,

            umaps=[10,2],
            scale=False,
            pca = 20,

            titles = "ABCDEFGHIJK",
            debug =  False,
            make_even=True):

        self.data= adatas
        self.titles = titles
        self.even = make_even
        self.debug=debug


        # preprocess
        self.preprocess(selector, selectgenes,
                        {'mean': meanexp, 'bins':bins,'plot': visual_ftsel})


        # do dimred
        self.dimz = dimensions.dimension_reduction(self.data,scale,False,PCA=pca,umaps=umaps)


        # set data
        if umaps:
            for i,d in zip(umaps,self.dimz):
                self.__dict__[f"d{i}"] = d
        else:
            self.dx = self.dimz

        #self.sort_cells(umaps)
        return self


    def sort_cells(self,umaps=[]):
        #assert False, "not implemented"
        for i in range(len(self.data)-1):
            hung, _ = u.hungarian(self.dx[i],self.dx[i+1])
            self.data[i+1].X = self.data[i+1].X[hung[1]]
            if umaps:
                for x in umaps:
                    self.__dict__[f"d{x}"][i+1] = self.__dict__[f"d{x}"][i+1][hung[1]]
            else:
                self.dx[i+1] = self.dx[i+1][hung[1]]


    def preprocess(self, selector, selectgenes, selectorargs):

        self.data = preprocess.basic_filter(self.data)  # min_counts min_genes
        self.data = preprocess.normlog(self.data)

        if selector == 'natto':
            genes = [preprocess.getgenes_natto(d, selectgenes,title, **selectorargs)
                     for d,title in zip(self.data, self.titles)]
        else:
            genes = [sc.pp.highly_variable_genes(d, n_top_genes=selectgenes) for d in self.data]

        self.data = preprocess.unioncut(genes, self.data)
        self.data = preprocess.make_even(self.data)

    def __init__(self):
        self.debug = False
