import matplotlib
matplotlib.use('module://matplotlib-sixel')

from natto import input
#a,b = input.loadimmune(pathprefix = '/home/ubuntu/repos/HungarianClustering/',subsample=250)
a,b = input.load3k6k(subsample=300,seed =3,pathprefix='/home/ubuntu/repos/HungarianClustering')

from natto import process
data  = process.Data().fit([a,b], visual_ftsel=False)
data.sort_cells()

from natto.process import cluster
y = cluster.gmm_1(data.d10[0])


from natto.process.cluster import k2means
y,e, labels, probas = k2means.multitunnelclust(data.d10,y)

from natto.out import draw
draw.cmp2(y,y,*data.d2)




'''
from natto.input import preprocessing as pp
import natto.process as p
from natto.out.quality import rari_score as score
from natto.process import k2means
from natto.out import draw


def prepare(a1,a2,quiet=True,debug_ftsel=False,clust = lambda x,y: p.gmm_2(x,y,nc=15), **kwargs):
    data = pp.Data()
    data.fit(a1,a2,quiet=quiet, debug_ftsel = debug_ftsel, **kwargs)
    data.sort_cells()
    data.utilclustlabels = clust(*data.dx)
    return data

def similarity(data):
    sim, randscore = score(*data.utilclustlabels, *data.dx)
    return sim


def tunnelclust(data, write_labels_to_data=False):
    labels, outliers = k2means.simulclust(*data.dx, data.utilclustlabels[0])
    data.tunnellabels = labels
    data.tunneloutliers = outliers
    return labels, outliers


def drawpair(data, tunnellabels = False):
    if tunnellabels:
        lab = data.tunnellabels.copy()
        lab[data.tunneloutliers] = -1
        draw.cmp2(lab,lab,*data.d2)
    else:
        draw.cmp2(*data.utilclustlabels,*data.d2)

'''



