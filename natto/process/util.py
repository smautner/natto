from lapsolverc import solve_dense
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances


class spacemap():
    # we need a matrix in with groups of clusters are the indices, -> map to integers
    def __init__(self, items):
        self.itemlist = items
        self.integerlist = list(range(len(items)))
        self.len = len(items)
        self.getitem = { i:k for i,k in enumerate(items)}
        self.getint = { k:i for i,k in enumerate(items)}


import numpy as np
def cleanlabels(asd):
    # asd is a list of label-lists

    items = np.unique(np.vstack(asd))
    s= spacemap(items)

    return [[s.getint[e] for e in li  ] for li in asd]


def hungarian(X1, X2, debug = False,metric='euclidean'):
    # get the matches:
    distances = pairwise_distances(X1,X2, metric=metric)
    #distances = ed(X1, X2)

    #if solver != 'scipy':
    #    row_ind, col_ind = linear_sum_assignment(distances)
    row_ind,col_ind = solve_dense(distances)

    if debug:
        x = distances[row_ind, col_ind]
        num_bins = 100
        print("hungarian: debug hist")
        plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
        plt.show()

    return (row_ind, col_ind), distances



