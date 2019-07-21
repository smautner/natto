
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from matplotlib_venn import *
def umap(X,Y, reducer = None,title="No title",acc : "y:str_description"={}, black = None):
    assert reducer != None  , "give me a reducer"
    plt.title(title)
    
    a=1
    b=0
    c=.5
    col={
        100:(a,a,b),
        101:(a,b,a),
        102:(b,a,a),
        103:(a,b,b),
        104:(b,a,b),
        105:(b,b,a),
        106:(c,c,b),
        107:(c,b,c),
        108:(b,c,c),
        109:(a,c,c),
        110:(c,a,c),
        111:(c,c,a)
    }
    col.update( {a-102:b for a,b in col.items()}  )

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
    plt.legend()
    plt.show()
    return reducer


def venn(one,two: 'boolean array', labels :"string tupple"):
    selover = [a and b for a,b in zip (one,two)]
    comb = sum(selover)
    v = venn2(subsets = {'10': sum(one)-comb, '01': sum(two)-comb, '11': comb}, set_labels = labels)
    plt.show()
