from collections import defaultdict
from natto import hungutil as hu
import numpy as np
from sklearn import  mixture
from natto.hungutil import spacemap 

class priorizedgmm(mixture.GaussianMixture):
    '''gmm where i set the initial class labels'''
    def _initialize_parameters(self, X, random_state):
        n_samples = X.shape[0] 
        resp = np.zeros((n_samples, self.n_components))
        
        for idd, label in zip(range(n_samples),self.labels):
            if label > -1: resp[idd,label] =1 

        self._initialize(X,resp)

    def fit(self, X, y=None):
        self.labels = y
        self.fit_predict(X, y)
        return self



def cluster(a,b,ca,cb, debug=False,normalize=True,draw=lambda x,y:None, maxsteps=10, gmmiter=3, numclust='max'):
    ro,co,dists = hu.hungarian(a,b)
    for i in range(maxsteps):
        '''
        the plan is this: 
            0. do maxsteps times:
            1. get class transfered class labels
            2. gmmiter steps of gmm 
            3. do the same for the other set
        '''
        ncb = len(np.unique(cb)) 
        nca = len(np.unique(ca))
        nc = max(nca,ncb) if numclust == 'max' else 0

        p=priorizedgmm(n_components=nc or ncb ,max_iter= gmmiter, random_state=45)
        p.labels = transferlabels(ro,co,dists,ca,cb, draw=draw, debug=debug, numclust=numclust) 
        cbold= np.array(cb)
        cb = p.fit_predict(b)

        p=priorizedgmm(n_components=nc or nca,max_iter= gmmiter, random_state=45)
        p.labels = transferlabels(co,ro,dists,cb,ca, reverse=True, draw=draw, debug=debug, numclust=numclust)
        caold= np.array(ca)
        ca = p.fit_predict(a)
        if debug: draw(ca,cb)
        

        # NEW DEBUGGING TO FIX MAXSTEP 

        print (i,all(ca == caold), all(cb == cbold ))
        print (ca[ca != caold])
        if all(ca == caold) and all(cb == cbold ):
            break

    return ca,cb, None


def transferlabels(ro,co,dists,ca,cb, reverse=False, draw= lambda x,y:None, debug = False, numclust='asd'): 
    
    # return labels in b such that matching labels cells have the same label
    

    # make a dict: classinA:[connections in b with distance]
    di = defaultdict(list)
    for a,b in zip(ro,co):
        di[ca[a]].append( ( dists[a,b] if not reverse else dists[b,a] ,b)  )
        #di[ca[a]].append( b  )
    answer = np.ones(len(cb), dtype=np.int)*-1
    

    # if a has more classes thanB, we delete the clustr (amd assign the cell somewhere else)  in A
    scores = [ (np.mean( [ d for d,_ in items  ]) , label )   for label, items in di.items() ]
    scores.sort()
    okclasses = [ label for _, label in  scores[:len(np.unique(cb))]  ]
    # print ("okclasses", okclasses)

    asd = spacemap(okclasses)
    for label, items in di.items():
        if label in okclasses or numclust=='max':
            answer[[ i for _,i in items ]] = label if numclust=='max' else asd.getint[label]

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
    






    

