#from lapsolver import solve_dense
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
import numpy as np

class spacemap():
    # we need a matrix in with groups of clusters are the indices, -> map to integers
    def __init__(self, items):
        self.itemlist = items
        self.integerlist = list(range(len(items)))
        self.len = len(items)
        self.getitem = { i:k for i,k in enumerate(items)}
        self.getint = { k:i for i,k in enumerate(items)}

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
    '''
    from time import time
    now = time()
    from lapjv import lapjv
    row_ind, col_ind, _ = lapjv(distances)
    print(f"  {time() - now}s"); now =time()
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(distances)
    print(f"  {time() - now}s"); now =time()
    from lapsolver import solve_dense
    row_ind,col_ind = solve_dense(distances)
    print(f"  {time() - now}s"); now =time()
    '''
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(distances)

    if debug:
        x = distances[row_ind, col_ind]
        num_bins = 100
        print("hungarian: debug hist")
        plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
        plt.show()

    return (row_ind, col_ind), distances



