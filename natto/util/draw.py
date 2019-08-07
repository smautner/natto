from itertools import permutations
import random
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from matplotlib_venn import *
def umap(X,Y, reducer = None,title="No title",acc : "y:str_description"={}, black = None):
    assert reducer != None  , "give me a reducer"
    plt.title(title)
    
    
    
    
    colors = list(permutations([0,.25,.5,.75,1],3))
    random.seed(4) #making shuffle consistent
    random.shuffle(colors)
    col = { i-2:e for i,e in enumerate(colors)}
    
    col.update( {a+100:b for a,b in col.items()}  )

    Y=np.array(Y)
    
    
    size= max( int(4000/Y.shape[0]), 1)
    
    if type(black) != type(None): 
        embed = reducer.transform(black)
        plt.scatter(embed[:, 0], embed[:, 1],c=[(0,0,0) for e in range(black.shape[0])], 
                    s=size,
                    label='del')
    
    embed = reducer.transform(X)
    for cla in np.unique(Y):
        plt.scatter(embed[Y==cla, 0], embed[Y==cla, 1],
                    color=col[cla],
                    s=size,
                    label= "%s %s" % (str(cla),acc.get(cla)))
    plt.axis('off')
    plt.legend(markerscale=4)
    plt.show()
    return reducer


def venn(one,two: 'boolean array', labels :"string tupple"):
    selover = [a and b for a,b in zip (one,two)]
    comb = sum(selover)
    v = venn2(subsets = {'10': sum(one)-comb, '01': sum(two)-comb, '11': comb}, set_labels = labels)
    plt.show()