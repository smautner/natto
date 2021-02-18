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
