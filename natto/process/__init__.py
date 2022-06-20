from natto.process import preprocess
import scanpy as sc
from natto.process import dimensions
from natto.process import util as u
from lmz import *

class Data():
    def fit(self,adataList,
            selector='natto',
            donormalize=True,
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
        self.donormalize=donormalize

        if selector == 'preselected':
            self.preselected_genes = self.data[0].preselected_genes

        # preprocess
        self.preprocess(selector, selectgenes,
                        {'mean': meanexp, 'bins':bins,'plot': visual_ftsel}, savescores=True)


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

        shapeofdataL = [x.shape for x in self.data]

        ### Normalize and filter the data
        self.data = preprocess.normfilter(self.data, self.donormalize)

        if selector == 'natto':
            if self.selectslice == 'last':
                genes, scores = Transpose([preprocess.getgenes_natto(self.data[-1],selectgenes, self.titles[-1], **selectorargs)]*len(self.data))
            else:
                genes,scores = Transpose([preprocess.getgenes_natto(d, selectgenes,title, **selectorargs)
                         for d,title in zip(self.data, self.titles)])
        elif selector == 'preselected':
            genes = np.array([[True if gene in self.preselected_genes else False for gene in x.var_names] for x in self.data])
            scores = genes.as_type(int)
        else:
            hvg_df = [np.array(sc.pp.highly_variable_genes(d, n_top_genes=selectgenes, flavor=selector, inplace=False)) for d in self.data]
            genes = [x['highly_variable_genes'] for x in hvg_df]
            if selector == 'seurat_v3':
                scores = [x['variances_norm'] for x in hvg_df]
            else:
                scores = [x['dispersions_norm'] for x in hvg_df]

        self.data = preprocess.unioncut(scores, selectgenes, self.data)
        self.genes = genes
        self.genescores = scores
        if self.even:
            self.data = preprocess.make_even(self.data)

        print("preprocess:")
        for a,b in zip(shapeofdataL, self.data):
            print(f"{a} -> {b.shape}")

