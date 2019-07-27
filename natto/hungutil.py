import basics as b
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


################
# maniulate distance matrix
#############
def discount_cluster(class_even,class_odd, distances, multiplier=.95):
    one,two = distances.shape
    for i in range(one):
        for j in range(two):
            if class_even[i] == class_odd[j]:
                distances[i,j] = distances[i,j]*multiplier
    return  distances

def test_discount_cluster():
    import numpy as np
    even = np.array([0]*5+[1]*3)
    odd = np.array([0]*4+[1]*3)
    row_ind = [1,7,2,3]
    col_ind = [2,6,5,4]
    dist = np.ones((8,7))
    print(discount_cluster(even,odd, dist, multiplier=.95))


###################
# purity meassure for matched clusters
#################
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




########################
# finding map between clusters 
#####################


class spacemap():
    def __init__(self, items):
        self.getitem = { i:k for i,k in enumerate(items)}
        self.getint = { k:i for i,k in enumerate(items)}

def find_clustermap_hung(Y1,Y2, hungmatch, debug=False):
    row_ind, col_ind = hungmatch
    pairs = zip(Y1[row_ind],Y2[col_ind])
    pairs = Counter(pairs) # pair:occurances
    
    # normalize for cluster size
    clustersizes= Counter(Y1) # class:occurence 
    clustersizes2 = Counter(Y2) # class:occurence 

    y1map = spacemap(clustersizes.keys())
    y2map = spacemap(clustersizes2.keys())

    #normpairs = { k:1-(float(v)/clustersizes[k[0]]) for k,v in pairs.items()} # pair:relative_occur
    normpairs = { k:-v for k,v in pairs.items()} # pair:relative_occur

    canvas = np.zeros( (len(clustersizes), len(clustersizes2)),dtype=float )


    for (a,b),v in normpairs.items():
        canvas[y1map.getint[a],y2map.getint[b]] = v
    
    row_ind, col_ind = linear_sum_assignment(canvas)
    
    if debug: # draw heatmap # for a good version of this check out notebook 10.1
        #print (canvas,y1map.getitem,y2map.getitem)
        
        # sort the canvas so that hits are on the diagonal
        sorting = sorted(zip(col_ind, row_ind, [y1map.getitem[r]for r in row_ind]))
        col_ind, row_ind, xlabels= list(zip(*sorting))
        # some rows are unused by the matching,but we still want to show them:
        rest= list(set(y1map.getitem.values())-set(row_ind) )
        canvas = canvas[list(row_ind)+rest]
        
        y1 = [y2map.getitem[c] for c in  col_ind]
        df = DataFrame(canvas)
        sns.heatmap(df,xticklabels=xlabels,yticklabels=y1, annot=True)
        plt.xlabel("Cluster ID data set arg1",size=15)
        plt.ylabel("Cluster ID data set arg2",size=15)
        plt.show()
    return dict(zip([y1map.getitem[r] for r in row_ind],[y2map.getitem[c] for c in  col_ind]))


def getcombos(alist): 
    '''returns all combinations'''
    return [e  for i in range(1,4) for e in list(combinations(alist,i))]


def get_min_subcost( itemsa, itemsb, costs ):
    subcosts = [ costs(a,b) for a in getcombos(itemsa) for b in getcombos(itemsb)]
    return min(subcosts)
    
    
def find_multi_clustermap_hung_optimize(pairs, clustercombos,clustercombos2,y1map,y2map, clustersizes1,clustersizes2,debug):
    # fill the cost matrix for the N:N set matches 
    # normalize: div the number of elements in 1 and 2 
    sumvalues = lambda keys,data: sum( [data[key] for key in keys  ])
    canvas = np.zeros( (len(clustercombos), len(clustercombos2)),dtype=float )
    for clusters_a in clustercombos:
        for clusters_b in clustercombos2:
            canvas[y1map.getint[clusters_a],y2map.getint[clusters_b]] = \
                -1 * sum([ pairs[c,d] for c in clusters_a for d in clusters_b ])  \
                /float( sumvalues(clusters_a,clustersizes1) + sumvalues(clusters_b,clustersizes2))

    if debug:
        debug_canvas = np.zeros( (len(clustercombos), len(clustercombos2)),dtype=float )
        for clusters_a in clustercombos:
            for clusters_b in clustercombos2:
                debug_canvas[y1map.getint[clusters_a],y2map.getint[clusters_b]] = \
                    -1 * sum([ pairs[c,d] for c in clusters_a for d in clusters_b ])  


    # MATCH 
    row_ind, col_ind = linear_sum_assignment(canvas)
    # MATCH 
    row_ind, col_ind = linear_sum_assignment(canvas)
    
    # no

    clustersets1 = [y1map.getitem[r] for r in row_ind]
    clustersets2 = [y2map.getitem[c] for c in  col_ind]
    costs  = [canvas[r][c] for r,c in zip(row_ind,col_ind)]
    cost_by_clusterindex = lambda x,y : canvas[y1map.getint[x],y2map.getint[y]] # cost accessed by cluster indices
    subcosts = [ get_min_subcost(y1map.getitem[r],  y2map.getitem[c], cost_by_clusterindex) for r,c in zip(row_ind,col_ind) ]
    
    if debug: # draw heatmap # for a good version of this check out notebook 10.1
        df = DataFrame(canvas)
        sns.heatmap(df,yticklabels=decorate(y1map.getitem),xticklabels=decorate(y2map.getitem), square=True)
        plt.show()
        #pprint.pprint(list (zip( clustersets1, clustersets2, costs, subcosts  ) ))
        df = DataFrame(canvas[:len(clustersizes1),:len(clustersizes2)])
        sns.heatmap(df,annot=True,yticklabels=clustersizes1.keys(),xticklabels=clustersizes2.keys(), square=True)
        plt.show()
        plt.subplots(figsize=(5,5))
        df = DataFrame(debug_canvas[:len(clustersizes1),:len(clustersizes2)])
        sns.heatmap(df,annot=True,yticklabels=clustersizes1.items(),xticklabels=clustersizes2.items(), square=True)
        plt.show()
        pprint.pprint( [ (y1,y2,cost,subcost) for y1,y2,cost,subcost in zip( clustersets1, clustersets2, costs, subcosts  ) if cost <= subcost ])
    return costs,subcosts, clustersets1,clustersets2


def cheapen_pairs(pairs, matches, factor):
    costchanges=defaultdict(int)
    for a,b in matches: # a and b are tupples
        for aa in a: 
            for bb in b:
                costchanges[(aa,bb)] += pairs[(aa,bb)]*factor
    for k,v in costchanges.items():
        pairs[k]+=v
    return pairs


def nodup(a):
    l = [zz for z in a for zz  in z]
    return len(l) == len(set(l))

def collisionfree(matches):
    a,b = list(zip(*matches))
    return nodup(a) and nodup(b)

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




        
    costs,subcosts,clustersets1,clustersets2 = find_multi_clustermap_hung_optimize(pairs,clustercombos,
                                                       clustercombos2,
                                                       y1map,
                                                       y2map,
                                                       clustersizes1,
                                                       clustersizes2,debug)
    while False: # looping was a bad idea, actually it worked a little... but i have a better 1 for now
        matches = [ (a,b) for (a,b,c,d) in zip( clustersets1, clustersets2, costs, subcosts  ) if c<=d ]
        if collisionfree(matches):
            break
        else:
            pairs = cheapen_pairs(pairs,matches, .5)
            cost,subcost,clustersets1,clustersets2 = find_multi_clustermap_hung_optimize(pairs,
                                                       clustercombos,
                                                       clustercombos2,
                                                       y1map,
                                                       y2map,
                                                       clustersizes1,
                                                       clustersizes2,debug)

    #return [ (y1,y2) for y1,y2,cost,subcost in zip( clustersets1, clustersets2, costs, subcosts  ) if subcost >= cost ]
    return [ (y1,y2) for y1,y2,cost,subcost in zip( clustersets1, clustersets2, costs, subcosts  ) if cost <= subcost ]

def decorate(items):
    l = range(len(items))
    return [' '.join(map(str,items[i]))for i in l ]



def test_find_clustermap():
    import numpy as np
    X= np.array(range(10))
    X=np.vstack((X,X)).T
    Y1= np.array([0]*3+[1]*3+[2]*4)
    Y2= np.array([1]*3+[2]*2+[0]*5)
    print ( find_clustermap_hung(X,X,Y1,Y2))


#######################
#
#######################

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



#####################
# does this still work?
###################
def distance_stats(row_ind,col_ind, class_even, class_odd, distances , printfails = False):
    pairs = list(zip(row_ind, col_ind))
    #pairs = zip(col_ind,row_ind)
    print ("WRONG:")
    print (sum([ 1 for a,b in pairs
        if class_even[a]!=class_odd[b]]))
    if printfails:
        print([ (a,b) for a,b in pairs if class_even[a]!=class_odd[b]])

    fails_i,fails_j = list(zip(*[ (a,b) for a,b in pairs
        if class_even[a]!=class_odd[b]]))
    asi_i,asi_j = list(zip(*[ (a,b) for a,b in pairs
        if class_even[a]==class_odd[b]]))
    fehl = distances[fails_i,fails_j]
    asi = distances[asi_i,asi_j]
    print ("mean,std of all distances",distances.mean(),distances.std())
    print ("mean,std of failures",fehl.mean(), fehl.std())
    print ("mean,std of correct asignments",asi.mean(), asi.std())
    sns.distplot(asi, hist=False, rug=False, label="assigned correct")
    sns.distplot(fehl, hist=False, rug=True, label="assigned false")
    sns.distplot(distances.ravel(), hist=False, rug=False, label="all distances")
    plt.legend(loc='upper right')

def test_distance_stats():
    """the drawing fails but that is ok"""
    import numpy as np
    even = np.array([0]*5+[1]*3)
    odd = np.array([0]*4+[1]*3)
    row_ind = [1,7,2,3]
    col_ind = [2,6,5,4]
    dist = np.zeros((10,10))
    distance_stats(row_ind, col_ind,even, odd, dist, printfails=True)


