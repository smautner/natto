# %%


import numpy as np
import matplotlib.pyplot as plt
import umap
from natto.out.quality import spacemap
from natto.out import draw
from natto import input
import seaborn as sns
from lmz import *
from ubergauss import tools
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
# %%
# 100 DATA..
#######################

def get_labels(labelpath, names = []):
    '''
    returns: 0-indexed labels,  and dictionary to get the str-label
    '''
    if not names:
        names = input.get100names(labelpath)

    labs = [n[:5] for n in names]
    items = np.unique(labs)
    s= spacemap(items)
    nulabels = [s.getint[e] for e in labs  ]
    nulabels=np.array(nulabels)
    return nulabels, s.getitem


def drawclustermap(data, intlabels, labeldict, ncol = 5):
    '''
    see better clustermap
    '''
    # draw
    colz = [draw.col[x] for x in intlabels]
    g=sns.clustermap(data,row_colors = colz, col_colors = colz)

    # ???
    for label in labeldict.keys():
        g.ax_col_dendrogram.bar(0, 0, color=draw.col[label],
                                label=labeldict[label], linewidth=0)

    # remove annoying ticks
    g.ax_heatmap.tick_params(right=False, bottom=False)
    g.ax_heatmap.set(xticklabels=[])
    g.ax_heatmap.set(yticklabels=[])
    # possition of legend
    g.ax_col_dendrogram.legend( ncol=ncol, bbox_to_anchor=(1,-4.1), fontsize= 18)
    return g

def betterclustermap(distances, labels, ncol = 5):
    '''
    why? compared to drawclustermap:
        - make squareform so we dont see the warning -> no way
        - do spacemap stuff internally so we can just pass a list of labels -> ok
        - *maybe* remove one dendro trees :) -> also not as easy
    '''

    mysm = tools.spacemap(labels)
    intlabels, labeldict = [mysm.getint[x] for x in labels],mysm.getitem


    # draw
    colz = [draw.col[x] for x in intlabels]
    '''
    tried to suppress the warning,, but its no good :/
    #sq = squareform(distances)
    #link = linkage(1-sq, method='average')
    #g=sns.clustermap(distances,row_colors = colz, col_colors = colz, row_linkage = link, col_linkage= link)
    '''
    g=sns.clustermap(distances,row_colors = colz, col_colors = colz)


    # ???
    for label in labeldict.keys():
        g.ax_col_dendrogram.bar(0, 0, color=draw.col[label],
                                label=labeldict[label], linewidth=0)

    # remove annoying ticks
    g.ax_heatmap.tick_params(right=False, bottom=False)
    g.ax_heatmap.set(xticklabels=[])
    g.ax_heatmap.set(yticklabels=[])
    # possition of legend
    g.ax_col_dendrogram.legend( ncol=ncol, bbox_to_anchor=(1,-4.1), fontsize= 18)
    return g



def plot(data,labels = [], save ='not implemented'):
    assert save == 'not implemented'
    intlabels, labeldict  = get_labels(None, names =labels)
    drawclustermap(data,intlabels,labeldict)
    plt.show()
    plt.close()


from sklearn.cluster import AgglomerativeClustering as AG
from sklearn.metrics import adjusted_rand_score as ra

def score(distance_matrix, labels, nc_range):
    """
    so we get the distance matrix for the 100x100 or whatever,
    then we choose max( rand(labels, agglo(data,n_clust)  ) vor all n_clust ) to score the data
    """
    l,_ =  get_labels(None, labels) # -> turn labels to int

    Lpredictions = [ AG(n_clusters= n_clust, linkage='ward').fit(distance_matrix).labels_ for n_clust in nc_range]

    #for e in Lpredictions: print(f" {e=}")

    scorez = Map(lambda z: ra(l,z), Lpredictions)
    print(f" score: {max(scorez)}")
    print(f"real labels: {l} {Zip(nc_range,scorez)=}")
    return max(scorez)



