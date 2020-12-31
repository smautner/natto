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
