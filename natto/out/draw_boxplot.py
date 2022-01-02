import numpy as np
from matplotlib import pyplot as plt
from natto.out import draw 

def get_testdata():
    return np.random.randint(0, 100, size=(10, 10, 100))

class grid_boxplot(): 
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
    d = data.shape[0]
    t = grid_boxplot(dim=(d,d))
    for i in range(d):
        for j in range(d):
            if i<=j:
                t.draw(data[i][j])
                plt.ylim((0,1))
            else:
                t.next()
    plt.tight_layout()
    plt.show()
 

def test_boxes():
    data = get_testdata()
    boxes(data)