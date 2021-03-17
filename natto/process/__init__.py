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
            make_even=True,
            sortfield=-1):

        self.data= adatas
        self.titles = titles
        self.even = make_even
        self.debug=debug


        # preprocess
        self.preprocess(selector, selectgenes,
                        {'mean': meanexp, 'bins':bins,'plot': visual_ftsel})


        # do dimred
        self.projections = dimensions.dimension_reduction(self.data,scale,False,PCA=pca,umaps=umaps)


        if pca:
            self.PCA = self.projections[0]

        if sortfield >=0: 
            self.sort_cells(sortfield)
        

        if umaps:
            for x,d in zip(umaps,self.projections[int(pca==True):]):
                self.__dict__[f"d{x}"] = d


        return self


    def sort_cells(self,projection_id = 1):
        # loop over data sets
        for i in range(len(self.data)-1):
            hung, _ = u.hungarian(self.projections[projection_id][i],self.projections[projection_id][i+1])
            self.data[i+1].X = self.data[i+1].X[hung[1]]

            # loop over projections
            for x in range(len(self.projections)):
                self.projections[x][i+1] = self.projections[x][i+1][hung[1]]

                


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
