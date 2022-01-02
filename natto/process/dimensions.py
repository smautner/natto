

import umap
import scanpy as sc
import numpy as np
from sklearn import decomposition

def dimension_reduction(adatas, scale, zero_center, PCA, umaps, joint_space=True):

    # get a (scaled) dx
    if scale or PCA:
         adatas= [sc.pp.scale(adata, zero_center=False, copy=True,max_value=10) for adata in adatas]
    dx = [adata.to_df().to_numpy() for adata in adatas]
    if joint_space == False:
        return disjoint_dimension_reduction(dx, PCA, umaps)


    res = []


    if PCA:
        pca = decomposition.PCA(n_components=PCA)
        pca.fit(np.vstack(dx))
        #print('printing explained_variance\n',list(pca.explained_variance_ratio_))# rm this:)
        dx = [ pca.transform(e) for e in dx  ]
        res.append(dx)

    for dim in umaps:
        assert 0 < dim < PCA
        res.append(umapify(dx,dim))

    return res


def umapify(dx, dimensions):
    mymap = umap.UMAP(n_components=dimensions,
                      n_neighbors=10,
                      random_state=1337).fit(np.vstack(dx))
    return [mymap.transform(a) for a in dx]


def disjoint_dimension_reduction(dx, PCA, umaps):
    res = []
    if PCA:
        pcaList = [decomposition.PCA(n_components=PCA) for x in dx]
        for pca, x in zip(pcaList, dx):
            pca.fit(np.vstack(x))
        #print('printing explained_variance\n',list(pca.explained_variance_ratio_))# rm this:)
        dx = [ pca.transform(e) for pca, e in zip(pcaList,dx)  ]
        res.append(dx)

    for dim in umaps:
        assert 0 < dim < PCA
        res.append(umapify(dx,dim))


    return res



