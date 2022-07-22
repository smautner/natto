from lmz import Map,Zip,Filter,Grouper,Range,Transpose

import numpy as np
from natto.optimize.x40 import assign
from sklearn.semi_supervised import LabelSpreading
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import adjusted_rand_score
from ubergauss import tools
def score1nn(y1,y2,Xlist):
    '''
    y1 is what we are supposed to predict :)
    '''
    clf = KNeighborsClassifier(n_neighbors=1).fit(Xlist[1], y2)
    l1  = clf.predict(Xlist[0])
    return adjusted_rand_score(y1,l1)


def hungonly(y1,y2,Xlist):
    return adjusted_rand_score(y1,y2)


def labelprobonly(y1,y2,Xlist):
    y1len = Xlist[0].shape[0]
    lp_model = LabelSpreading( alpha = .2, max_iter = 30,n_jobs=1)
    startlabels = np.hstack((np.full_like(y1,-1),y2))
    args = np.vstack(Map(tools.zehidense,Xlist)), startlabels
    lp_model.fit(*args)
    l1,l2= np.split( lp_model.transduction_,2)
    return adjusted_rand_score(y1,l1)


