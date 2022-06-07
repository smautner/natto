
import matplotlib
matplotlib.use('module://matplotlib-sixel')
import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from assign import *
from sklearn.semi_supervised import LabelSpreading
'''
circle example from sklearn.. lets see if we can also diffuse some labels
'''


from structout.intlistV2 import binning




def plotc(X,labels):
    for l in np.unique(labels):
        plt.scatter(
            X[labels == l, 0],
            X[labels == l, 1],
            #color="navy",
            marker="s",
            lw=0,
            label=str(l),
            s=15,
        )

    plt.legend(scatterpoints=1, shadow=False, loc="upper right")
    plt.show()
    plt.close()


def circles():

    '''
    build testdata and vizualize
    '''
    n_samples = 200
    X, y = make_circles(n_samples=n_samples, shuffle=False)
    y[y==1] = -1
    y[y==0] = binning(np.arange(sum(y==0)), 6, False)
    plotc(X,y)


    '''
    diffuse and vizualize
    '''
    lp_model = LabelSpreading(kernel = lambda x,y: mykernel(100,2,x,y))
    lp_model.fit(X,y)
    plotc(X, lp_model.transduction_)


def testkernel():
    x1len = 4
    neighbors = 1
    X = np.array([[x,x] for x in Range(x1len)+Range(2)])
    g,d = mykernel(x1len=x1len, neighbors=neighbors, X=X, _=X, return_graph=True)
    print(g)
    print(d)
    res  = mykernel(x1len=x1len, neighbors=neighbors, X=X, _=X, return_graph=False)
    print(res)

    lp_model = LabelSpreading(kernel = lambda x,y: mykernel(x1len,neighbors,x,y))
    lp_model.fit(X,[-1,-1,-1,-1,0,2])
    print(f"{lp_model.transduction_=}")


