def antidiv2_hung(a,b,costs): # variation on antidiv. bad.
    matrix = np.array([[ costs((aa,),(bb,))  for aa in a] for bb in b])
    r,c = linear_sum_assignment(matrix)
    return sum([matrix[rr,cc] for (rr,cc) in zip(r,c)])/float(np.sum(matrix))

def antidiv3_hung(a,b,pairs): # variation on antidiv. bad.
    matrix = np.array([[ pairs[aa,bb]  for aa in a] for bb in b])
    r,c = linear_sum_assignment(matrix)
    return sum([matrix[rr,cc] for (rr,cc) in zip(r,c)])/float(np.sum(matrix))

def _gini(a,b,costs): # unused
    return gini(np.array([ costs((aa,),(bb,)) for aa in a for bb in b   ]))

def _dist_to_mean(a,b,costs ): # unused
    stuff = np.array([ costs((aa,),(bb,)) for aa in a for bb in b   ])
    stuff -= np.mean(stuff)
    stuff *=stuff
    return np.mean(stuff)



def mapclusters(X,X2,Yh,Y2h) -> 'new Yh and {class-> quality}':

    # return  new assignments; stats per class (to be put in legend); overall accuracy
    hungmatch = hungarian(X,X2)
    clustermap = find_clustermap_hung(Yh,Y2h, hungmatch)
    
    class_acc = qualitymeassure(Yh,Y2h,hungmatch,clustermap)
        
    # renaming according to map
    Yh = [clustermap.get(a,10) for a in Yh]

    return Yh, {clustermap.get(int(a)):"%.2f" % b for a,b in class_acc[:-1]},class_acc[-1][1] 



def get_min_subcost_too_strict(itemsa, itemsb, costs):  # this is too strict on multi hits
    canvas = np.zeros( (len(itemsa), len(itemsb)),dtype=float )
    for i,a in enumerate(itemsa):
        for j,b in enumerate(itemsb):
            canvas[i,j]= costs((a,),(b,))
            
    return matrix_cost_estimate(canvas)



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


def gini(array): # dowsnt work here
    """Calculate the Gini coefficient of a numpy array."""
    # by oligiaguest
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 0.0000001
    array = np.sort(array)
    index = np.arange(1,array.shape[0]+1)
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))




def matrix_cost_estimate(matrix): 
    return sum([matrix[r,c] for (r,c) in zip(*linear_sum_assignment(matrix))])




def greedymarriage(matrix): 
    rind = list(range(matrix.shape[0]))
    cind = list(range(matrix.shape[1]))
    s= [ (matrix[r,c],r,c) for r  in rind for c in cind]
    s.sort()
    for (v,r,c) in s: 
        if r in rind and c in cind:
            yield r,c
            rind.remove(r)
            cind.remove(c)

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

    
    
# old merging algo... not sufficient because multiple rounds should be required
def upwardmerge(re):
    def match(e,l): 
        # e ist tupple, l ist list of sets
        # if anything from e is in a set of l, we return the index of l or -1
        for i,s in enumerate(l):
            if any([ clu in s for clu in e]):
                return i
        return -1
    
    def add(tu,sett):
        for e in tu:
            sett.add(e)
        
    l1,l2 = [],[]
    
    for e1,e2 in re: 
        merged=False
        z=match(e1,l1)
        if z > -1: 
            add(e1,l1[z])
            merged=True
        z=match(e2,l2)
        if z > -1: 
            add(e2,l2[z])
            merged=True
        
        if not merged:
            l1.append(set(e1))
            l2.append(set(e2))
    
    return list(zip(map(tuple,l1),map(tuple,l2)))