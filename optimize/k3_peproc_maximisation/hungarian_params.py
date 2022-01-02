
'''
we want to know under which preprocessing the hungarian works best...
so we load the 3/6k , preprocess, hungarian -> adjusted rand index
'''

from lmz import *
from natto.process import util
from sklearn.metrics import adjusted_rand_score as ari
from natto import process
from natto import input

d2,d10,pc,o = [],[],[],[]
x = Range(20,100,5)
pca_var = Range(70,90,5)


# 3k6k
loader = lambda: input.load3k6k(pathprefix= '/home/ubuntu/repos/HungarianClustering',subsample = 1000, seed = 3)
labels = 'labels'

# 100 DATASET
if True:
	datasetname = input.get71names()[4]
	loader1 = lambda seed: input.load100(datasetname,subsample=1000,path='/home/ubuntu/repos/natto/natto/data',seed=seed, remove_unlabeled=True)
	loader = lambda: [loader1(5), loader1(11)]
	labels = 'true'





#data = loader()
#zomg = process.Data().fit(list(data), visual_ftsel=False,umaps = [], pca = 300, make_even=True)
#exit()

for pca in x:
    # load data
    data  =  loader()
    # preprocess
    zomg = process.Data().fit(list(data), visual_ftsel=False,umaps = [2,10], pca = pca, make_even=True)
    l1,l2 = [a.obs[labels] for a in zomg.data]
    def rate(stuff):
        a,b = stuff
        (i,j),_ = util.hungarian(a,b)
        r=ari(l1[i], l2[j])
        print(r)
        return [r]
    print("#############################")
    pc+=rate(zomg.PCA)
    d2+=rate(zomg.d2)
    d10+=rate(zomg.d10)
    o+=rate([a.X for a in zomg.data])

print([pc,d2,d10,o,x])


