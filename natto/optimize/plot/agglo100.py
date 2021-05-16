# %% 


import numpy as np
import matplotlib.pyplot as plt 
import umap
from natto.out.quality import spacemap
from natto.out import draw 
from natto import input

# %%
# 100 DATA.. 
#######################

# load distance matrix for 100x100 and the labels
def get_matrix(median=False, path = "/home/ikea/data/100x.lst"): 
    alldata = eval(open(path,'r').read())
    alldata=np.array(alldata)
    if not median: 
        return alldata
    alldata[alldata==-1]=np.nan
    z = np.nanmean(alldata,axis=2)
    return z 

# labels
def get_labels():
    '''
    returns: 0-indexed labels,  and dictionary to get the str-label '''
    names = input.get100names("/home/ikea/data/data")
    labs = [n[:5] for n in names]
    items = np.unique(labs)
    s= spacemap(items)
    nulabels = [s.getint[e] for e in labs  ] 
    nulabels=np.array(nulabels)
    return nulabels, s.getitem


# %% we umap projection... 
def scatter(u, lab, labdi):
    for cid in di.keys():
        X = u[lab == cid]
        plt.scatter(X[:,0], X[:,1],label=labdi[cid], color=draw.col[cid])
    plt.legend()
    plt.show()

if False:
    z = get_matrix(median=True)
    labels, labelnames = get_labels()
    fit = umap.UMAP(n_neighbors=3)
    u = fit.fit_transform(z)
    scatter(u, labels, labelnames)





# %% agglo plot 
import seaborn as sns 


z = get_matrix(median=True)
lab, dic = get_labels()


def drawclustermap(z, labels, dic, ncol = 5):
    # draw
    colz = [draw.col[x] for x in labels]
    g=sns.clustermap(z,row_colors = colz, col_colors = colz)

    # ???
    for label in dic.keys():
        g.ax_col_dendrogram.bar(0, 0, color=draw.col[label],
                                label=dic[label], linewidth=0)

    # remove annoying ticks
    g.ax_heatmap.tick_params(right=False, bottom=False)
    g.ax_heatmap.set(xticklabels=[])
    g.ax_heatmap.set(yticklabels=[])
    # possition of legend
    g.ax_col_dendrogram.legend( ncol=ncol, bbox_to_anchor=(1,-4.1), fontsize= 18)
    plt.show()

drawclustermap(z,lab, dic)

# %% 

from sklearn.cluster import AgglomerativeClustering as AG
from sklearn.metrics import adjusted_rand_score as ra
geneZ = get_matrix(median=True,path = "/home/ikea/projects/data/natto/natto_latest/distance_genes.ev")
# note: using the median will reduce performance
jacca = lambda x: x/(1600-x)
J = jacca(geneZ)

def myscore(m,lab):
    s= []
    for nclu in range(5,14):
        mo=AG(n_clusters= nclu).fit(m)
        s.append(ra(mo.labels_, lab)) 
    print(max(s))

drawclustermap(J,lab, dic)
myscore(J,lab)
z = get_matrix(median=True) # median makes it better
myscore(z,lab)

#myscore(geneZ[:,:,1],lab)
# get clusters by setting (0..alot) in agglo -> can calculate rand index -> report max
# quality of clustering for each cluster -> are we improvin on small clusters? -> plot clustersize vs performancegain
#geneZ = get_matrix(median=True,path = "/home/ikea/projects/data/natto/natto_latest/distance_genes.ev")
#drawclustermap(geneZ,lab, dic)


# %%
# here i will draw the clustermap for 
# repeated sampling 
##############
#distance_fake_multi.ev

import seaborn as sns 


def get_data2(slide,rep_id=1):
    DATAA  = eval(open("/home/ikea/projects/data/natto/distance_fake_multi_nuplacenta.ev",'r').read())
    DATAA = np.array(DATAA)
    DATAA[DATAA==-1]=0
    DATA = DATAA[:,:,1]
    return DATA


def get_labels2():
    LABELS  = eval(open("/home/ikea/projects/data/natto/distance_fake_multi_labels.ev",'r').read())
    LABELS  = [l[:5] for l in LABELS] 
    
    items = np.unique(LABELS)
    s= spacemap(items)
    nulabels = [s.getint[e] for e in LABELS] 
    nulabels=np.array(nulabels)

    return nulabels, s.getitem

z= get_data2(1)
labels, dic = get_labels2()
drawclustermap(z, labels, dic)


# %%
# cutting out so we have one celltype 

z = get_matrix(median=False)
labels, labelnames = get_labels()

for typ, v in labelnames.items():
    onecelltype = z[labels == typ]
    onecelltype = onecelltype[:,labels == typ,:5]
    data = np.hstack([onecelltype[:,:,i] for i in range(5)])
    instances = sum(labels == typ)
    drawclustermap(data, list(range(instances))*5 , {i:f"{v}-{i}"for i in range(instances)}, ncol=4)

# %%


# cutting such that we have multiple celltypes
z = get_matrix(median=False)
labels, labelnames = get_labels()
import random
from collections import defaultdict
r = defaultdict(list)
for i,e in enumerate(labels):
    r[e].append(i)

selection = np.array([ random.choice(r[k])  for k in r.keys() ])

onecelltype = z[selection]
onecelltype = onecelltype[:,selection,:5]
data = np.vstack([onecelltype[:,:,i] for i in range(5)])
instances = len(selection)
drawclustermap(data, list(range(instances))*5 , labelnames)

# %%