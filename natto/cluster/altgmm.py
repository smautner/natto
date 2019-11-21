from collections import defaultdict
from natto import hungutil as hu
import numpy as np
from sklearn import  mixture


class priorizedgmm(mixture.GaussianMixture):
    '''gmm where i set the initial class labels'''
    def _initialize_parameters(self, X, random_state):
        resp = np.zeros((n_samples, self.n_components))
        self._initialize(X,resp)
        label=self.labels
        resp[np.arange(n_samples), label] = 1
    def fit(self, X, y=None):
        self.labels = y
        self.fit_predict(X, y)
        return self



def cluster(a,b,ca,cb, debug=False,normalize=True,draw=lambda x,y:None, maxsteps=10, gmmiter=3):
    ro,co,dists = hu.hungarian(a,b)
    for i in range(maxsteps):
        '''
        the plan is this: 
            0. do maxsteps times:
            1. get class transfered class labels
            2. gmmiter steps of gmm 
            3. do the same for the other set
        '''
        p=priorizedgmm(n_components=np.unique(cb),max_iter= gmmiter)
        p.labels = transferlabels(ro,co,dists,ca,cb, draw=draw, debug=debug) 
        cb = p.fit_predict(b)

        p=priorizedgmm(n_components=np.unique(ca),max_iter= gmmiter)
        p.labels = transferlabels(co,ro,dists,cb,ca, reverse=True, draw=draw, debug=debug) 
        draw(ca,cb)

    return ca,cb, None


def transferlabels(ro,co,dists,ca,cb, reverse=False, draw= lambda x,y:None, debug = False): 
    
    # return labels in b such that matching labels cells have the same label
    
    di = defaultdict(list)
    for a,b in zip(ro,co):
        di[ca[a]].append( ( dists[a,b] if not reverse else dists[b,a] ,b)  )

    
    answer = np.ones(len(cb))*-1

    for label, items in di.items():
        answer[items] = label

    '''
    for clas, tlist in di.items():
        tlist.sort(reverse=False)
        cut = int(len(tlist)*.5)
        tlist1 =  [ b for a,b in  tlist[:cut]] 
        arglist =[ b for a,b in  tlist[cut:] ] 
        conlist+=arglist
        mustlink += [(bb,bbb) for bb in tlist1 for bbb in tlist1]
    '''

    if debug:
        print("this should highlight the change:")
        if not reverse:
            draw( ca ,answer)
        if reverse:
            draw( answer ,ca)
    return answer
    






    

