from natto.process import preprocess
import scanpy as sc
from natto.process import dimensions
from natto.process import util as u
from lmz import *

class Data():
    def fit(self,adataList,
            selector='natto',
            selectgenes=800,
            selectslice='all',

            meanexp = (0.015,4),
            bins = (.25,1),
            visual_ftsel=True,

            umaps=[10,2],
            scale=False,
            pca = 20,
            joint_space = True,

            titles = "ABCDEFGHIJK",
            make_even=True,
            sortfield=-1):
        '''
        sortfield = 0  -> use adata -> TODO one should test if the cell reordering works when applied to anndata
        '''
        self.data= adataList
        self.titles = titles
        self.even = make_even
        self.selectslice = selectslice

        if selector == 'preselected':
            self.preselected_genes = self.data[0].preselected_genes

        # preprocess
        self.preprocess(selector, selectgenes,
                        {'mean': meanexp, 'bins':bins,'plot': visual_ftsel})


        # do dimred
        self.projections = [[ d.X for d in self.data]]+dimensions.dimension_reduction(self.data,scale,False,PCA=pca,umaps=umaps, joint_space=joint_space)



        if pca:
            self.PCA = self.projections[1]

        if sortfield >=0:
            self.sort_cells(sortfield)


        if umaps:
            for x,d in zip(umaps,self.projections[int(pca>0)+1:]):
                self.__dict__[f"d{x}"] = d
        return self


    def sort_cells(self,projection_id = 0):
        # loop over data sets
        for i in range(len(self.data)-1):
            hung = self.hungarian(projection_id,i,i+1)
            self.data[i+1] = self.data[i+1][hung[1]]
            #self.data[i+1].X = self.data[i+1].X[hung[1]]
            # loop over projections
            for x in range(len(self.projections)):
                self.projections[x][i+1] = self.projections[x][i+1][hung[1]]
    #
    def hungarian(self,data_fld,data_id, data_id2):
            hung, _ = u.hungarian(self.projections[data_fld][data_id],self.projections[data_fld][data_id2])
            return hung


    def preprocess(self, selector, selectgenes, selectorargs, savescores = False):

        shapeofdataL = [x.shape for x in self.data]

        self.data = preprocess.basic_filter(self.data)  # min_counts min_genes
        self.data = preprocess.normlog(self.data)

        if selector == 'natto':
            if self.selectslice=='last':
                #genes,scores = Transpose([preprocess.getgenes_natto(self.data[-1], selectgenes,title, **selectorargs)
                         #for d,title in zip(self.data, self.titles)])     
                genes, scores = Transpose([preprocess.getgenes_natto(self.data[-1],selectgenes, self.titles[-1], **selectorargs)]*len(self.data))           
            else:
                genes,scores = Transpose([preprocess.getgenes_natto(d, selectgenes,title, **selectorargs)
                         for d,title in zip(self.data, self.titles)])
            if savescores:
                self.genescores = scores
        elif selector == 'preselected':
            genes = np.array([[True if gene in self.preselected_genes else False for gene in x.var_names] for x in self.data])
        else:
            genes = [sc.pp.highly_variable_genes(d, n_top_genes=selectgenes) for d in self.data]

        self.data = preprocess.unioncut(genes, self.data)
        self.genes=genes
        self.data = preprocess.make_even(self.data)

        print("preprocess:")
        for a,b in zip(shapeofdataL, self.data):
            print(f"{a} -> {b.shape}")












import numpy as np
class Data_DELME():
    """
        so we set the number of genes to 800,
        this guy selects more than 1 gene-set.
    """
    def fit(self,adatas,
            selector='natto',
            selectgenes=2000,

            meanexp = (0.015,4),
            bins = (.25,1),
            visual_ftsel=True,

            umaps=[10,2],
            scale=False,
            pca = 20,

            titles = "ABCDEFGHIJK",
            make_even=True,
            sortfield=-1):

        self.data= adatas
        self.titles = titles
        self.even = make_even


        # preprocess
        self.preprocess(selector, selectgenes,
                        {'mean': meanexp, 'bins':bins,'plot': visual_ftsel})

        # do dimred
        self.projections = [ dimensions.dimension_reduction(d,scale,False,PCA=pca,umaps=umaps) for d in self.data]

        return self


    def sort_cells(self,projection_id = 1):
        # loop over data sets
        for i in range(len(self.data)-1):
            hung, _ = u.hungarian(self.projections[projection_id][i],self.projections[projection_id][i+1])
            self.data[i+1] = self.data[i+1][hung[1]]
            #self.data[i+1].X = self.data[i+1].X[hung[1]]

            # loop over projections
            for x in range(len(self.projections)):
                self.projections[x][i+1] = self.projections[x][i+1][hung[1]]



    def preprocess(self, selector, selectgenes, selectorargs, savescores = False):

        self.data = preprocess.basic_filter(self.data)  # min_counts min_genes
        self.data = preprocess.normlog(self.data)

        genes,scores = Transpose([preprocess.getgenes_natto(d, selectgenes,title, **selectorargs)
                 for d,title in zip(self.data, self.titles)])

        genelist = [[self.select(x,g,s) for g,s in zip(genes,scores)] for x in range(100,2000,100) ]

        self.data = [preprocess.unioncut(genes, self.data) for genes in genelist]
        self.data = [ preprocess.make_even(d) for d in self.data]

    def select(self,num, genes, scores):
       # there is a gene array and a score array..

       scores[np.logical_not(genes)] = -2
       selecthere = np.argsort(scores)[:num]
       r=np.full(len(genes), 0)
       r[selecthere] = 1
       return r

