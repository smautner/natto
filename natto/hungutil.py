import basics as b
import math
import pprint
import networkx as nx
from itertools import combinations, permutations
from collections import Counter
from sklearn.metrics.pairwise import euclidean_distances as ed
#from eden import display as eden
from matplotlib import pyplot as plt
import numpy as np
from collections import  defaultdict
import random

from scipy.optimize import linear_sum_assignment
from lapsolver import solve_dense

from natto.util import draw
    
def hungarian(X1,X2, solver='scipy'):
    # get the matches:
    distances = ed(X1,X2)         
    if solver== 'scipy':
        row_ind, col_ind = linear_sum_assignment(distances)
    else:
        row_ind,col_ind = solve_dense(distances)
    return row_ind, col_ind



########################
# finding map between clusters 
#####################
class spacemap():
    # we need a matrix in with groups of clusters are the indices, -> map to integers
    def __init__(self, items):
        self.itemlist = items
        self.integerlist = list(range(len(items)))
        self.len = len(items)
        self.getitem = { i:k for i,k in enumerate(items)}
        self.getint = { k:i for i,k in enumerate(items)}



def getcombos(alist): 
    '''returns all combinations, aka powerset'''
    # [1,2,3] => [1],[2],[3],[1,2],[2,3],[3,2],[1,2,3]
    return [e  for i in range(1,6) for e in list(combinations(alist,i))]


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




def upwardmerge(re):
    # convert to sets so we can easyly add clusterids
    re = [ (set(a),set(b)) for a,b in re]
   

    def set_in_dict(se,di):
        # check if element of set is already in the dict
        for e in se:
            if e in di:
                return di[e]
        return -1
    
    # loop to merge
    finished = False
    while not finished:
        
        # save which items we have seen 
        d1 = {}
        d2 = {}
        # loop current solution
        for i  in range(len(re)):
            s1,s2 = re[i]
            
            # if we see something that was there before, merge and break
            j = max(set_in_dict(s1,d1),set_in_dict(s2,d2))
            if j > -1:
                for e in s1: re[j][0].add(e)
                for e in s2: re[j][1].add(e)
                del re[i]
                break
                
            #else: put in dict of observed items 
            d1.update({e:i for e in s1})
            d2.update({e:i for e in s2})
        else:
            # if we dont break, no collisions were detected
            finished =True
            
    return [ (tuple(a),tuple(b)) for a,b in re ]
            



def find_multi_clustermap_hung_optimize(pairs,y1map,y2map, clustersizes1,clustersizes2,debug, method='scipy'):
    # fill the cost matrix for the N:N set matches 
    # normalize: div the number of elements in 1 and 2
    # do matching
    sumvalues = lambda keys,data: sum( [data[key] for key in keys  ])
    canvas = np.zeros( (y1map.len, y2map.len ),dtype=float )
    for clusters_a in y1map.itemlist:
        for clusters_b in y2map.itemlist:
            numoverlap =   float(sum([ pairs[c,d] for c in clusters_a for d in clusters_b ]))
            sizeab = float( (sumvalues(clusters_a,clustersizes1) + sumvalues(clusters_b,clustersizes2) ))            
            v = -2*numoverlap / sizeab
            canvas[y1map.getint[clusters_a],y2map.getint[clusters_b]] = v
            #canvas[y1map.getint[clusters_a],y2map.getint[clusters_b]] = v if v < -.15 else 0.0
    row_ind, col_ind = linear_sum_assignment(canvas) if method=='scipy' else  solve_dense(canvas)

    
    # clustersets -> translate from the indices of the matrix back to cluster_ids 
    # calculate costs and sub-costs for filtering 
    clustersets1 = [y1map.getitem[r] for r in row_ind]
    clustersets2 = [y2map.getitem[c] for c in  col_ind]
    costs  = [canvas[r][c] for r,c in zip(row_ind,col_ind)]
    cost_by_clusterindex = lambda x,y : canvas[y1map.getint[x],y2map.getint[y]] # cost accessed by cluster indices
    subcosts = [ get_min_subcost(y1map.getitem[r],  y2map.getitem[c], cost_by_clusterindex) for r,c in zip(row_ind,col_ind) ]
    
    if debug: # draw heatmap # for a good version of this check out notebook 10.1
        
        if debug:
            debug_canvas = np.zeros( (y1map.len, y2map.len ),dtype=float )
            for clusters_a in y1map.itemlist:
                for clusters_b in y2map.itemlist:
                    debug_canvas[y1map.getint[clusters_a],y2map.getint[clusters_b]] = \
                        -1 * sum([ pairs[c,d] for c in clusters_a for d in clusters_b ]) 
        
        # HEATMAP normalized 
        draw.heatmap(canvas,y1map,y2map)
        # HEATMAP total 
        draw.heatmap(debug_canvas,y1map,y2map)
        
        # costs before diversity meassure
        '''
        print ("#"*80)
        pprint.pprint( [ (y1,y2,cost,
                          antidiversity(y1,y2,cost_by_clusterindex)
                          #antidiv2_hung(y1,y2,cost_by_clusterindex),
                          #antidiv3_hung(y1,y2,pairs)
                         ) for y1,y2,cost,subcost in zip( clustersets1, clustersets2, costs, subcosts  ) if cost <= subcost ])
        '''
    #return costs,subcosts, clustersets1,clustersets
    
    
    # filter by 1.low subcost, 2. stuff that looks like a diagonal matrix
    result = [ (y1,y2) for y1,y2,cost,subcost in zip( clustersets1, clustersets2, costs, subcosts  ) if cost <= subcost and antidiversity(y1,y2,cost_by_clusterindex) > .3]
    
    result = upwardmerge(result)
    if debug: pprint.pprint(result)
    return result


def find_multi_clustermap_hung(Y1,Y2, hungmatch, debug=False, method='scipy'):
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
                                                       clustersizes2,debug,method=method)


    
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


def duplicates(lot):
    seen = {}
    for tup in lot: 
        for e in tup:
            if e in seen:
                return False
            else:
                seen[e]=True
    return True

def multimapclusters(X,X2,Yh,Y2h, debug=False,method = 'lapsolver') -> 'new Yh,Y2h, {class-> quality}, global quality':
    # return  new assignments; stats per class (to be put in legend); overall accuracy
    hungmatch = hungarian(X,X2, solver=method)
    clustermap = find_multi_clustermap_hung(Yh,Y2h, hungmatch,debug=debug, method=method) 
    
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



def leftoverassign(canvas, y1map,y2map, row_ind, col_ind):
    # assign leftover classes to the best fit 
    diff_a = list( set(y1map.integerlist) - set(row_ind))
    diff_b = list( set(y2map.integerlist) - set (col_ind))
    
    while len(diff_a) > 0:
        element = diff_a.pop()
        row_ind = np.concatenate((row_ind,[element]))
        col_ind= np.concatenate((col_ind,[np.argmin(canvas[element,:])]))
   
    while len(diff_b) > 0:
        element = diff_b.pop()        
        col_ind = np.concatenate((col_ind,[element]))
        row_ind= np.concatenate((row_ind,[np.argmin(canvas[:,element])]))
    return row_ind, col_ind
   

def make_canvas_and_spacemaps(Y1,Y2,hungmatch,normalize=True):
    row_ind, col_ind = hungmatch
    pairs = zip(Y1[row_ind],Y2[col_ind])
    pairs = Counter(pairs) # pair:occurance
    
    # normalize for cluster size
    clustersizes  = Counter(Y1) # class:occurence 
    clustersizes2 = Counter(Y2) # class:occurence 

    y1map = spacemap(clustersizes.keys())
    y2map = spacemap(clustersizes2.keys())

    if normalize:
        normpairs = { k:float(-v)/float(clustersizes[k[0]]+clustersizes[k[1]]) for k,v in pairs.items()} # pair:relative_occur
    else:
        normpairs = { k:-v for k,v in pairs.items()} # pair:occur
    
    canvas = np.zeros( (y1map.len, y2map.len),dtype=float )
    for (a,b),v in normpairs.items():
        canvas[y1map.getint[a],y2map.getint[b]] = v
    
    return y1map, y2map, canvas

# annotate LOSSes for assigning weirdo clusters ,,,, for ROw and COlumn
def rocoloss(a,b,y1map,y2map,canvas):
    a=y1map.getint[a]
    b=y2map.getint[b]
    amax = np.min(canvas[:,b])-canvas[a,b]
    offender_a = y1map.getitem[np.argmin(canvas[:,b])]
    bmax = np.min(canvas[a,:])-canvas[a,b]
    offender_b = y2map.getitem[np.argmin(canvas[a,:])]
    return (amax,bmax,offender_a,offender_b)
        
def find_clustermap_one_to_one(Y1,Y2, hungmatch, debug=False, normalize=True):
    
    row_ind, col_ind = hungmatch
    y1map,y2map,canvas = make_canvas_and_spacemaps(Y1,Y2,hungmatch)
    
    # MAKE ASSIGNMENT 
    row_ind, col_ind = linear_sum_assignment(canvas)
        
    # assign leftovers to best hit
    row_ind, col_ind = leftoverassign(canvas, y1map,y2map, row_ind, col_ind)

    result =  [ ((a,),(b,),rocoloss(a,b,y1map,y2map,canvas)) for a,b in zip([y1map.getitem[r] for r in row_ind],[y2map.getitem[c] for c in  col_ind])]
    
    if debug: 
        draw.heatmap(canvas,y1map,y2map)
        print(" clsuter in first set, cluster in second set,  (loss in row, loss in col, reason for row loss, reason for col loss)")
        pprint.pprint(result)
        
    return upwardmerge([ (a,b) for a,b,c in result]) #[ ((a,),(b,)) for a,b in zip([y1map.getitem[r] for r in row_ind],[y2map.getitem[c] for c in  col_ind])]





def multimapclusters_single (X,X2,Yh,Y2h, debug=False,method = 'lapsolver', normalize=True) -> 'new Yh,Y2h, {class-> quality}, global quality':
    
    # Find match
    hungmatch = hungarian(X,X2, solver=method)
    clustermap = find_clustermap_one_to_one(Yh,Y2h, hungmatch, debug=debug, normalize=normalize) 
    
    # lets make sure that all the class labels are corrected
    # first we need the classtupples of the clustermap to point to the same class
    y1map = spacemap([a for a,b in clustermap])
    y2map = spacemap([b for a,b in clustermap])
    # seperating y1 and y2 
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










from natto.cluster.simple import predictgmm
def recluster(data,Y,problemcluster):
    data2 = data[Y==problemcluster]
    yh = predictgmm(2,data2)
    Y[Y==problemcluster] = yh+np.max(Y)+1
    return Y
    


##############
# we do 1:1 and do some splitting until all is good 
###############

def find_clustermap_one_to_one_and_split(Y1,Y2, hungmatch, data1,data2, debug=False, normalize=True):
    row_ind, col_ind = hungmatch
    y1map,y2map,canvas = make_canvas_and_spacemaps(Y1,Y2,hungmatch)
    
    # MAKE ASSIGNMENT 
    row_ind, col_ind = linear_sum_assignment(canvas)
        
    # assign leftovers to best hit
    row_ind, col_ind = leftoverassign(canvas, y1map,y2map, row_ind, col_ind)

    result =  [ ((a,),(b,),rocoloss(a,b,y1map,y2map,canvas)) for a,b in zip([y1map.getitem[r] for r in row_ind],[y2map.getitem[c] for c in  col_ind])]
    
    if debug: 
        draw.heatmap(canvas,y1map,y2map)
        print(" clsuter in first set, cluster in second set,  (loss in row, loss in col, reason for row loss, reason for col loss)")
        pprint.pprint(result)
        

    split = [ (c,e,1) for a,b,(c,d,e,f) in result if  c < -.1]+[ (d,f,2) for a,b,(c,d,e,f) in result if  d < -.1]

    if not split: # no splits required
        translator =  {b:a for (a,),(b,),c in result} 
        return Y1, [translator[e] for e in Y2], {a:b for a,b,c in result} 
    
    split.sort()
    _,cluster,where = split[0]
    if where ==1: 
        Y1 = recluster(data1,Y1,cluster)
    elif where ==1:
        Y2 = recluster(data2,Y2,cluster)
    
    
    return find_clustermap_one_to_one_and_split(Y1,Y2,hungmatch,debug,normalize)
    
