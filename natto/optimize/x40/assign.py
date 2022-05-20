import x40_dmp as dmp
from natto import input, process
'''
1. load the dict a => b,c OK
2. load the data           OK
3. joint pp (either b or b,c) OK

4. diffuse the real labels from b,, assign labels blabla

5. eval

'''

neighbors = dmp.neighs()


def getzedata(li,neighs=1,numcells=1500, seed = 31337):
    return [input.load100(x,
                          path = "/home/ubuntu/repos/natto/natto/data",
                          seed = seed,
                          subsample=numcells) for x in li[:neighs+1]]

zedata = getzedata(neighbors[1], neighs = 1, numcells = 1500)

data = process.Data().fit(zedata,
            visual_ftsel=False,
            pca = 20,
            make_readcounts_even=True,
            umaps=[10,2],
            sortfield = 0,# real labels need to follow the sorting i think...
            make_even=True)

# plot adata.obs['true']

import matplotlib
matplotlib.use('module://matplotlib-sixel')
import natto.out.draw as draw
draw.cmp2(*[data.data[x].obs['true'] for x in [0,1]],*data.d2)


#



