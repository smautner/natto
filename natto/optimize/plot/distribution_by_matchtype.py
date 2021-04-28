# %%
import numpy as np 
from lmz import *
from matplotlib import pyplot as plt

# %%
# 100 DATA.. 
#######################

# load distance matrix for 100x100 and the labels
alldata = eval(open("/home/ikea/data/100x.lst",'r').read())
alldata=np.array(alldata)
alldata[alldata==-1]=np.nan
data = np.nanmean(alldata,axis=2)

# labels
from natto import input
names = input.get100names("/home/ikea/data/data")
labs = [n[:5] for n in names]






# find incides where stuff=same
idx =[[i,i] for i in range(len(data))]
f  = list(range(100))
same = data[f,f]

# indices where same label
lab = [ (i,j) for i,y in enumerate(labs) for j,z in enumerate(labs) if y==z and i!=j ]

typ = data[Transpose(lab)]
# rest

unrelated = [ (i,j) for i,y in enumerate(labs) for j,z in enumerate(labs) if i!=j ]

unrelated = data[Transpose(unrelated)]



# %%

X = np.arange(0,1,.01)


def plotline(X,data, label):
    
    def getval(x,d):
        return sum(data < x)/ len(data)
    y = [ getval(x,data) for x in X ]
    print(y)
    plt.plot(X,y, label=label)

plotline(X, same,'dataset')
plotline(X, typ,'celltype')
plotline(X, unrelated,'unlrelated')


plt.xlabel("similarity")
plt.ylabel("coverage")
plt.legend()
# %%
