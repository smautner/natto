import basics as b
import math
import pprint
import networkx as nx
from itertools import combinations, permutations
from collections import Counter
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances as ed
import seaborn as sns
from eden import display as eden
from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame
from collections import  defaultdict
import random


    
def hungarian(X1,X2):
    # get the matches:
    distances = ed(X1,X2)         
    row_ind, col_ind = linear_sum_assignment(distances)
    return row_ind, col_ind







########################
# finding map between clusters 
#####################



class spacemap():
    # we need a matrix in with groups of clusters are the indices, -> map to integers
    def __init__(self, items):
        self.itemlist = items
        self.integerlist = list(range(items))
        self.len = len(items)
        self.getitem = { i:k for i,k in enumerate(items)}
        self.getint = { k:i for i,k in enumerate(items)}



def getcombos(alist): 
    '''returns all combinations, aka powerset'''
    # [1,2,3] => [1],[2],[3],[1,2],[2,3],[3,2],[1,2,3]
    return [e  for i in range(1,10) for e in list(combinations(alist,i))]


def get_min_subcost( itemsa, itemsb, costs ):  
    # given 2 sets of clusters, look at all combinations in the 2 powersets, return cost of min
    subcosts = [ costs(a,b) for a in getcombos(itemsa) for b in getcombos(itemsb)]
    return min(subcosts)

    

def antidiversity(a,b,costs):
    # if i matchh 2 set of clusters, 
    # ill have a matrix of len_a x len_b with all the intersectsions
    # i dont want sqrt(len_matrix) to be of high value and the rest 0
    # this function penalizes exactly that
    
    stuff = [ costs((aa,),(bb,)) for aa in a for bb in b   ]
    if len(stuff)<=1:
        return 1 
    stuff.sort()
    max_ = float(min(stuff))
    stuff = [s/max_ for s in stuff]
    cut = int(math.sqrt(len(stuff)))
    #cut = int(len(stuff)/2.0)
    s1 = stuff[1:cut]
    s2 = stuff[cut:]
    return ( cut-1 - sum(s1) + sum(s2) )/float(len(stuff)-1)

  
def find_multi_clustermap_hung_optimize(pairs,y1map,y2map, clustersizes1,clustersizes2,debug):
    # fill the cost matrix for the N:N set matches 
    # normalize: div the number of elements in 1 and 2
    # do matching
    sumvalues = lambda keys,data: sum( [data[key] for key in keys  ])
    canvas = np.zeros( y1map.len, y2map.len ,dtype=float )
    for clusters_a in y1map.itemlist:
        for clusters_b in y2map.itemlist:
            numoverlap =   sum([ pairs[c,d] for c in clusters_a for d in clusters_b ])  
            sizeab = float( (sumvalues(clusters_a,clustersizes1) + sumvalues(clusters_b,clustersizes2) ))            
            canvas[y1map.getint[clusters_a],y2map.getint[clusters_b]] = -2*numoverlap / sizeab
    row_ind, col_ind = linear_sum_assignment(canvas)

    
    # clustersets -> translate from the indices of the matrix back to cluster_ids 
    # calculate costs and sub-costs for filtering 
    clustersets1 = [y1map.getitem[r] for r in row_ind]
    clustersets2 = [y2map.getitem[c] for c in  col_ind]
    costs  = [canvas[r][c] for r,c in zip(row_ind,col_ind)]
    cost_by_clusterindex = lambda x,y : canvas[y1map.getint[x],y2map.getint[y]] # cost accessed by cluster indices
    subcosts = [ get_min_subcost(y1map.getitem[r],  y2map.getitem[c], cost_by_clusterindex) for r,c in zip(row_ind,col_ind) ]
    
    if debug: # draw heatmap # for a good version of this check out notebook 10.1
        
        if debug:
            debug_canvas = np.zeros( y1map.len, y2map.len ),dtype=float )
            for clusters_a in y1map.items:
                for clusters_b in y2map.items:
                    debug_canvas[y1map.getint[clusters_a],y2map.getint[clusters_b]] = \
                        -1 * sum([ pairs[c,d] for c in clusters_a for d in clusters_b ]) 
        
        df = DataFrame(canvas)
        #plt.subplots(figsize=(10,10))
        #sns.heatmap(df,annot=False,yticklabels=decorate(y1map.getitem),xticklabels=decorate(y2map.getitem), square=True)
        #plt.show()
        #pprint.pprint(list (zip( clustersets1, clustersets2, costs, subcosts  ) ))
        df = DataFrame(canvas[:len(clustersizes1),:len(clustersizes2)])
        #df=df.apply(lambda x: x/np.abs(np.min(x)))
        sns.heatmap(df,annot=True,yticklabels=clustersizes1.keys(),xticklabels=clustersizes2.keys(), square=True)
        plt.show()
        df = DataFrame(debug_canvas[:len(clustersizes1),:len(clustersizes2)])
        #df=df.apply(lambda x: x/np.abs(x.min()))
        sns.heatmap(df,annot=True,yticklabels=clustersizes1.items(),xticklabels=clustersizes2.items(), square=True)
        plt.show()
        pprint.pprint( [ (y1,y2,cost,subcost
                          #antidiv2_hung(y1,y2,cost_by_clusterindex),
                          #antidiv3_hung(y1,y2,pairs)
                         ) for y1,y2,cost,subcost in zip( clustersets1, clustersets2, costs, subcosts  ) ])
        print ("#"*80)
        pprint.pprint( [ (y1,y2,cost,
                          antidiversity(y1,y2,cost_by_clusterindex)
                          #antidiv2_hung(y1,y2,cost_by_clusterindex),
                          #antidiv3_hung(y1,y2,pairs)
                         ) for y1,y2,cost,subcost in zip( clustersets1, clustersets2, costs, subcosts  ) if cost <= subcost ])
        
    #return costs,subcosts, clustersets1,clustersets
    
    
    # filter by 1.low subcost, 2. stuff that looks like a diagonal matrix
    result = [ (y1,y2) for y1,y2,cost,subcost in zip( clustersets1, clustersets2, costs, subcosts  ) if cost <= subcost and antidiversity(y1,y2,cost_by_clusterindex) > .3]
    
    return upwardmerge(result)


def find_multi_clustermap_hung(Y1,Y2, hungmatch, debug=False):
    '''clustermatching allowing N:N matching'''

    

    # absolute matches in each pair of clusters (one from 1, one from 2)
    row_ind, col_ind = hungmatch
    pairs = zip(Y1[row_ind],Y2[col_ind])
    pairs = Counter(pairs) # pair:occurances
    

    # size of all clusters
    clustersizes1 = Counter(Y1)
    clustersizes2 = Counter(Y2)

    # get the powerset of clusternames
    clustercombos  = getcombos(Counter(Y1).keys()) # list 
    clustercombos2 = getcombos(Counter(Y2).keys())# list 

    # the elements of the powerset dont make good indices.. so we map them to continuous integers
    y1map = spacemap(clustercombos)
    y2map = spacemap(clustercombos2)

    
    return find_multi_clustermap_hung_optimize(pairs,  y1map,
                                                       y2map,
                                                       clustersizes1,
                                                       clustersizes2,debug)


    '''
    costs,subcosts,clustersets1,clustersets2 = find_multi_clustermap_hung_optimize(pairs,
                                                       y1map,
                                                       y2map,
                                                       clustersizes1,
                                                       clustersizes2,debug)
    while False: # looping was a bad idea, actually it worked a little... but i have a better 1 for now
        matches = [ (a,b) for (a,b,c,d) in zip( clustersets1, clustersets2, costs, subcosts  ) if c<=d ]
        if collisionfree(matches): # func was moved to -> bad.py
            break
        else:
            pairs = cheapen_pairs(pairs,matches, .5)# -> was movced to bad.py
            cost,subcost,clustersets1,clustersets2 = find_multi_clustermap_hung_optimize(pairs,
                                                       y1map,
                                                       y2map,
                                                       clustersizes1,
                                                       clustersizes2,debug)

    #return [ (y1,y2) for y1,y2,cost,subcost in zip( clustersets1, clustersets2, costs, subcosts  ) if subcost >= cost ]
    return [ (y1,y2) for y1,y2,cost,subcost in zip( clustersets1, clustersets2, costs, subcosts  ) if cost <= subcost ]
    '''
    
def decorate(items):
    l = range(len(items))
    return [' '.join(map(str,items[i]))for i in l ]



#######################
# fancy stuff for calling
#######################


def qualitymeassure(Y1,Y2,hungmatch,matches):
    '''
    Y1 and Y2: are the predicted classes for instances
    hungmatches: are the matches between Y1 and Y2 (between instances)
    matches: are the matches on class level  (e.g.: class1 -> class2)

    returns: [str:class,float:sum(correct in class)/#class )] 
    '''
    row_ind, col_ind = hungmatch
    # first  count correctly matched instances
    sumcorrect = defaultdict(int)
    for a,b in zip(row_ind, col_ind): 
        if Y1[a] in matches:
            if matches[Y1[a]] == Y2[b]:
                sumcorrect[Y1[a]]+=1 
    
    # lets make some stats from that 
    classes = Counter(Y1) # class: occurance
    return [ (a, float(sumcorrect[a])/count)   for a,count in classes.items()] + [('all', float(sum(sumcorrect.values())) / len(Y1) ) ]


def mapclusters(X,X2,Yh,Y2h) -> 'new Yh and {class-> quality}':

    # return  new assignments; stats per class (to be put in legend); overall accuracy
    hungmatch = hungarian(X,X2)
    clustermap = find_clustermap_hung(Yh,Y2h, hungmatch)
    
    class_acc = qualitymeassure(Yh,Y2h,hungmatch,clustermap)
        
    # renaming according to map
    Yh = [clustermap.get(a,10) for a in Yh]

    return Yh, {clustermap.get(int(a)):"%.2f" % b for a,b in class_acc[:-1]},class_acc[-1][1] 

def  duplicates(lot):
    seen = {}
    for tup in lot: 
        for e in tup:
            if e in seen:
                return False
            else:
                seen[e]=True
    return True

def multimapclusters(X,X2,Yh,Y2h, debug=False) -> 'new Yh,Y2h, {class-> quality}, global quality':
    # return  new assignments; stats per class (to be put in legend); overall accuracy
    hungmatch = hungarian(X,X2)
    clustermap = find_multi_clustermap_hung(Yh,Y2h, hungmatch,debug=debug) 
    
    # lets make sure that all the class labels are corrected
    # first we need the classtupples of the clustermap to point to the same class
    y1map = spacemap([a for a,b in clustermap])
    y2map = spacemap([b for a,b in clustermap])
    
    # setperating y1 and y2 
    # then mapping the classes i to the right new class
    y1names, y2names = list(zip(*clustermap))
    assert duplicates(y1names) , 'cluster assignments went wrong, call multimapclusters with debug'
    assert duplicates(y2names) , 'cluster assignments went wrong, call multimapclusters with debug ' 
    y1names = {i:y1map.getint[a] for a in y1names for i in a }
    y2names = {i:y2map.getint[a] for a in y2names for i in a }
   
    # now we only need to translate
    Yh = b.lmap(lambda x:y1names.get(x,-2) , Yh) 
    Y2h = b.lmap(lambda x:y2names.get(x,-2) , Y2h) 

    clustermap = {y1map.getint[a]:y2map.getint[b] for a,b in clustermap }
    class_acc = qualitymeassure( Yh ,Y2h , hungmatch,clustermap)

    return Yh,Y2h, {clustermap.get(a):"%.2f" % b for a,b in class_acc[:-1]},class_acc[-1][1] 


