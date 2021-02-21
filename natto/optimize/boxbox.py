# %% lets assume we have a 3d data struct..

import numpy as np
data  = np.random.randint(0, 100, size=(10, 10, 100))




d=10

# make a boxplot from the data
from matplotlib import pyplot as plt
from natto.out import draw 

class gryd(): 
    def __init__(self, dim=(3,3), size= 2):
        figs = (size*dim[1], size*dim[0])
        plt.figure( figsize=figs, dpi=100)
        self.i =0
        self.dim = dim
    
    def next(self): 
        self.i= self.i+1 
        plt.subplot(*self.dim,self.i)

    def draw(self, data): 
        self.next()
        plt.boxplot(data)



def boxes(data):
    t = gryd(dim=(d,d))
    for i in range(d):
        for j in range(d):
            if i<=j:
                t.draw(data[i][j])
            else:
                t.next()
    plt.tight_layout()
    plt.show()
boxes(data)


#  write median 
median = np.median(data,axis = 2)
print (median)
# write stretched median 

med2 = np.array(median)
for i in range(d):
    for j in range(d):
        if i!=j:
            med2[i,j]= med2[i,j]*(2/(median[i,i]+median[j,j]))

print(med2)



######
#  loading my res files:
######
import basics as ba
import os
def read_data(path='./res',ax0=9,ax1=9,Rep=20, mirror=False):
    alldata = np.zeros((ax0,ax1,Rep))
    for x in range(ax0):
        for y in range(ax1):
            for r in range(Rep):
                filename= f'{path}/{x}_{y}_{r}'
                if os.path.exists(filename):
                    data = ba.loadfile(filename)
                    alldata[x,y,r]= data
                    if mirror and x!=y:
                        alldata[y,x,r] = data
    return alldata


dist = read_data("",9,9,20,mirror=True)
dist = np.median(dist, axis =2 )
combi_perf = read_data("",9,9,20,mirror=True)
combi = np.median(combi_perf, axis =2 )

def normalize_dist(d):
    # input diag matrix 
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if i!=j:
                d1 = d[i,i]  +  d[j,j] 
                d[i,j] =  d[i,j]*(2/d1)  # or the other way araound

    return d


#################
# we want to do the ordering stuff like before
############

