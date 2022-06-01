import x40_dmp as dmp
from natto import input, process
from sklearn.metrics import adjusted_rand_score
import natto.out.draw as draw
import matplotlib
matplotlib.use('module://matplotlib-sixel')
import matplotlib.pyplot as plt

'''
1. load the dict a => b,c OK
2. load the data           OK
3. joint pp (either b or b,c) OK

4. diffuse the real labels from b,, assign labels blabla

5. eval

'''



def getzedata(li,neighs=1,numcells=1500, seed = 31337):
    datasetnames = li[:neighs+1]
    zedata= [input.load100(x,
                          path = "/home/ubuntu/repos/natto/natto/data",
                          seed = seed,
                          subsample=numcells) for x in datasetnames]

    zedata =  process.Data().fit(zedata,
            visual_ftsel=False,
            pca = 20,
            make_readcounts_even=True,
            umaps=[2],
            sortfield = 0,# real labels need to follow the sorting i think...
            make_even=True)

    truelabels = [zedata.data[x].obs['true'] for x in [0,1]]
    return datasetnames, zedata, truelabels



neighbors = dmp.neighs(draw=False)
plt.show()

for i in [10]:
    names, zedata, truelabels = getzedata(neighbors[i], neighs = 1, numcells = 1500)
    draw.cmp2(*truelabels,*zedata.d2)
    print(f"annotation score: {adjusted_rand_score(*truelabels)}")
    print(f"{names=}")

    # 2x confusion








