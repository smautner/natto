
import matplotlib
#matplotlib.use('module://matplotlib-sixel')
import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from natto.optimize.x40.assign import *
from sklearn.semi_supervised import LabelSpreading
'''
circle example from sklearn.. lets see if we can also diffuse some labels
'''


from structout.intlistV2 import binning



from matplotlib.rcsetup import cycler
def plotc(X,labels,fname=None, show = False):
    plt.figure(figsize=(6,6))
    mycycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#000']
    for l in np.unique(labels):
        plt.scatter(
            X[labels == l, 0],
            X[labels == l, 1],
            #color="navy",
            marker="s",
            c = mycycle[l],
            lw=0,
            label=str(l),
            s=15,
        )

    plt.legend(scatterpoints=1, shadow=False, loc="upper right")
    if fname:
        plt.savefig(fname)
    if show:
        plt.show()
        plt.close()


def circles():
    '''
    this tests the diffusion on a 2 circles.. outer and inner each representin a SC dataset

    ...build testdata and vizualize
    '''
    n_samples = 200
    X, y = make_circles(n_samples=n_samples, shuffle=False)
    y[y==1] = -1
    # the zeros are in the begining so we manyally overwrite a few more
    y[:30] = -1
    y[y==0] = binning(np.arange(sum(y==0)), 5, False)
    #plotc(X,y,'circle_1.png')
    plotc(X,y,'zomg3.svg')

    '''
    diffuse and vizualize
    '''
    lp_model = LabelSpreading(kernel = lambda x,y: mykernel(100,2,x,y))
    lp_model.fit(X,y)
    plotc(X, lp_model.transduction_,'zomg4.svg')


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






