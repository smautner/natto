import numpy as np
import pandas as pd
from natto.process import preprocess
import scanpy as sc
from scipy.sparse import csr_matrix
from natto.process import dimensions
from natto.process import util as u
from lmz import *

class Data():
    def fit(self,adataList,
            selector='natto',
            donormalize=True,
            min_counts=3,
            normTogether=False,
            selectgenes=800,
            selectslice='all',
            meanexp = (0.015,4),
            bins = (.25,1),

            titles="ABCDEFGHIJK",

            visual_ftsel=True,
            umaps=[10,2],
            scale=False,
            pca = 20,
            plotPCA=False,
            joint_space = True,
            make_even=True,
            sortfield=-1):
        '''
        sortfield = 0  -> use adata -> TODO one should test if the cell reordering works when applied to anndata
        '''
        self.donormalize=donormalize
        self.min_counts=min_counts
        self.normTogether=normTogether
        self.data= adataList
        self.titles = titles
        self.even = make_even
        self.selectslice = selectslice

        if selector == 'preselected':
            self.preselected_genes = self.data[0].preselected_genes

        # preprocess
        self.preprocess(selector, selectgenes,
                        {'mean': meanexp, 'bins':bins,'plot': visual_ftsel}, savescores=True)


        # do dimred
        self.projections = [[ d.X for d in self.data]]+dimensions.dimension_reduction(self.data,scale,False,PCA=pca,umaps=umaps, joint_space=joint_space, plotPCA=plotPCA)

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
        self.hungdist =[]
        for i in range(len(self.data)-1):
            hung, dist = self.hungarian(projection_id,i,i+1)
            self.hungdist.append(dist)
            #self.data[i+1] = self.data[i+1][hung[1]]
            #self.data[i+1].X = self.data[i+1].X[hung[1]]
            for x in range(len(self.projections)):
                self.projections[x][i+1] = self.projections[x][i+1][hung[1]]
                if x == 0:
                    self.data[i+1]= self.data[i+1][hung[1]]
    #
    def hungarian(self,data_fld,data_id, data_id2):
            hung, dist = u.hungarian(self.projections[data_fld][data_id],self.projections[data_fld][data_id2])
            return hung, dist[hung]


    def preprocess(self, selector, selectgenes, selectorargs, savescores = False):

        # shapeofdataL = [x.shape for x in self.data]

        ### Normalize and filter the data
        self.data = preprocess.normfilter(self.data, self.donormalize, min_counts=self.min_counts, normTogether=self.normTogether)

        if selector == 'natto':
            if self.selectslice == 'last':
                genes, scores = Transpose([preprocess.getgenes_natto(self.data[-1],selectgenes, self.titles[-1], **selectorargs)]*len(self.data))
            else:
                genes,scores = Transpose([preprocess.getgenes_natto(d, selectgenes,title, **selectorargs)
                         for d,title in zip(self.data, self.titles)])

        elif selector == 'preselected':
            selectgenes=len(self.preselected_genes)
            genes = [np.array([True if gene in self.preselected_genes else False for gene in x.var_names]) for x in self.data]
            scores = [x.astype(int) for x in genes]
            #scores = list(genes.astype(int))

        else:
            hvg_df = [sc.pp.highly_variable_genes(d, n_top_genes=selectgenes, flavor=selector, inplace=False) for d in self.data]
            genes = [np.array(x['highly_variable']) for x in hvg_df]
            if selector == 'seurat_v3':
                ### Best used for raw_count data
                scores = [np.array(x['variances_norm'].fillna(0)) for x in hvg_df]
            else:
                scores = [np.array(x['dispersions_norm'].fillna(0)) for i, x in enumerate(hvg_df)]



        self.data = preprocess.unioncut(scores, selectgenes, self.data)
        self.genes = preprocess.unioncut(scores, selectgenes, genes)
        self.genescores = preprocess.unioncut(scores, selectgenes, scores)
        if self.even:
            self.data = preprocess.make_even(self.data)

        # print("preprocess:")
        # for a,b in zip(shapeofdataL, self.data):
        #     print(f"{a} -> {b.shape}")



def annotate_genescores(adata, selector='natto',
        donormalize=True,
        nattoargs = {'mean':(0.015, 4),'bins':(.25, 1)},
        mingenes = 200,
        quiet = False,
        plot=False):

        incommingshape= adata.X.shape
        sc.pp.filter_cells(adata, min_genes=mingenes, inplace=True)
        okgenes = sc.pp.filter_genes(adata, min_counts=3, inplace=False)[0]
        if donormalize:
            sc.pp.normalize_total(adata, 1e4)
            sc.pp.log1p(adata)

        adata2 = adata.copy()
        adata = adata[:,okgenes]
        if selector == 'preselected':
            self.preselected_genes = self.data[0].preselected_genes

        if selector == 'natto':
                # "A" TODO
                genes, scores = preprocess.getgenes_natto(adata, 1000, 'A', plot=plot, **nattoargs)

        elif selector == 'preselected':
            genes = [True if gene in self.preselected_genes else False for gene in adata.var_names]
            scores = genes.as_type(int)

        else:
            hvg_df = sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor=selector, inplace=False)
            genes = np.array(hvg_df['highly_variable'])
            if selector == 'seurat_v3':
                ### Best used for raw_count data
                scores = np.array(hvg_df['variances_norm'].fillna(0))
            else:
                scores = np.array(hvg_df['dispersions_norm'].fillna(0))


        #fullscores = np.zeros(adata2.X.shape[1])
        fullscores = np.full(adata2.X.shape[1],np.NINF,np.float)
        fullscores[okgenes==1] = scores
        adata2.varm["scores"]=  fullscores
        adata2.varm['genes'] = okgenes
        #adata.varm["genes"] = genes ... lets decide later if we need this
        if not quiet:
            print(f"{incommingshape=}  => {adata.X.shape}")
        return adata2


class merge():
    def __init__(self, adatas, selectgenes = 800, make_even = True, pca = 20, umaps = [2],
            joint_space = False,
            sortfield = -1,
            titles = "ABCDEFGHIJKIJ"):

        shapesbevorepp= [a.X.shape for a in adatas]
        self.genescores = [a.varm['scores'] for a in adatas]
        self.geneab = [a.varm['genes'] for a in adatas]
        self.data  = preprocess.unioncut(self.genescores, selectgenes, adatas)

        # geneab = np.all(np.array(self.geneab), axis=0)
        # for i, d in enumerate(self.data):
        #     data[i] = d[:, geneab]

        if make_even:
            self.data = preprocess.make_even(self.data)


        print("preprocess:", end= '')
        for a,b in zip(shapesbevorepp, self.data):
            print(f"{a} -> {b.shape}")

        ######################
        self.titles = titles

         # do dimred
        self.projections = [[ d.X for d in self.data]]+dimensions.dimension_reduction(self.data,
                                                                False, # scale (will still scale if pca)
                                                                False, # will be ignored anyway
                                                                PCA=pca,
                                                                umaps=umaps,
                                                                joint_space=joint_space)

        if pca:
            self.PCA = self.projections[1]

        if sortfield >=0:
            self.sort_cells(sortfield)

        if umaps:
            for x,d in zip(umaps,self.projections[int(pca>0)+1:]):
                self.__dict__[f"d{x}"] = d


    def sort_cells(self,projection_id = 0):
        # loop over data sets
        self.hungdist =[]
        for i in range(len(self.data)-1):
            hung, dist = self.hungarian(projection_id,i,i+1)
            self.hungdist.append(dist)
            #self.data[i+1] = self.data[i+1][hung[1]]
            #self.data[i+1].X = self.data[i+1].X[hung[1]]
            for x in range(len(self.projections)):
                self.projections[x][i+1] = self.projections[x][i+1][hung[1]]
                if x == 0:
                    self.data[i+1]= self.data[i+1][hung[1]]
    #
    def hungarian(self,data_fld,data_id, data_id2):
            hung, dist = u.hungarian(self.projections[data_fld][data_id],self.projections[data_fld][data_id2])
            return hung, dist[hung]
