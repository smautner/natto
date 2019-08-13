from itertools import permutations
import random
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from umap import UMAP
import seaborn as sns

from matplotlib_venn import *

def umap(X,Y, reducer = None,title="No title",acc : "y:str_description"={}, black = None, show=True):
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
                    label= "%s %s" % (str(cla),acc.get(cla,'')))
    plt.axis('off')
    plt.legend(markerscale=4)
    if show: plt.show()
    return reducer


def cmp(Y1,Y2,X1,X2,title=('1','2'),red=None):    
    if not red:
        red = UMAP()
        red.fit(np.vstack((X1,X2)))
    plt.figure(figsize=(10,4))    
    ax=plt.subplot(121)
    umap(X1,Y1,red,show=False,title=title[0])
    ax=plt.subplot(122)
    umap(X2,Y2,red,show=False,title=title[1])
    plt.show()

def venn(one,two: 'boolean array', labels :"string tupple"):
    selover = [a and b for a,b in zip (one,two)]
    comb = sum(selover)
    v = venn2(subsets = {'10': sum(one)-comb, '01': sum(two)-comb, '11': comb}, set_labels = labels)
    plt.show()

def heatmap(canvas,y1map,y2map):
        # there is a version that sorts the hits to the diagonal in util/bad... 
        df = DataFrame(canvas[:y1map.len,:y2map.len])
        s= lambda y,x: [ y.getitem[k] for k in x]
        sns.heatmap(df,annot=True,yticklabels=y1map.itemlist,xticklabels=y2map.itemlist, square=True)
        plt.show()