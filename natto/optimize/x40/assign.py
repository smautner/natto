import x40_dmp as dmp
from natto import input, process
'''
1. load the dict a => b,c OK
2. load the data
3. joint pp (either b or b,c)
4. diffuse the real labels from b,, assign labels blabla
5. text how nice we do

'''

neighbors = dmp.neighs()


def getzedata(li,neighs=1,cells=1500):



zedata = getzedata(neighbors[1], neighs = 1, cells = 1500)



data = process.Data().fit(zedata,
            visual_ftsel=False,
            pca = 20,
            make_readcounts_even=True,
            umaps=[10,2],
            sortfield = 0,
            make_even=True)


def preprocess( repeats =7, ncells = 1500, out = 'data.dmp'):
    datasets = input.get40names()
    random.seed(43)
    loaders =  [ partial( input.load100,
                          data,
                          path = "/home/ubuntu/repos/natto/natto/data",
                          subsample=ncells)
                 for data in datasets]
    it = [ [loaders[i],loaders[j]] for i in Range(datasets) for j in range(i+1, len(datasets))]
    def f(loadme):
        a,b = loadme
        return [Data().fit([a(),b()],
            visual_ftsel=False,
            pca = 0,
            make_readcounts_even=True,
            umaps=[],
            sortfield = -1,
            make_even=True) for i in range(repeats)]
    res =  tools.xmap(f,it,32)
    tools.dumpfile(res,out)




