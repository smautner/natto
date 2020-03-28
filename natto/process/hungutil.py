import basics as ba
import time
import tabulate
import math
import pprint
from itertools import combinations
from collections import Counter
from sklearn.metrics.pairwise import euclidean_distances as ed
import numpy as np
from collections import  defaultdict
from scipy.optimize import linear_sum_assignment
from lapsolver import solve_dense
from natto.out import draw
import matplotlib.pyplot as plt

import umap
from natto.old.simple import predictgmm

#####
# first some utils
######
def hungarian(X1,X2, solver='scipy', debug = False):
    # get the matches:
    distances = ed(X1,X2)         
    if solver != 'scipy':
        row_ind, col_ind = linear_sum_assignment(distances)
    else:
        row_ind,col_ind = solve_dense(distances)

    if debug: 
        x = distances[row_ind, col_ind]
        num_bins = 100
        plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
        plt.show()
    return row_ind, col_ind, distances



class renamelog:
    def __init__(self,cla, name):
        self.rn = {c:c for c in cla}
        self.dataset=name # an id or name ;)

    def log(self,oldname, newname):
        # {2:1, 3:1}
        # input 2, 4    => {4:1}
        for o in oldname:
            for n in newname:
                self.rn[n] = self.rn[o]
            self.rn.pop(o)

    def total_rename(self,a):
        # a is a dictionary
        oldlookup = dict(self.rn)
        self.rn = { n:oldlookup[o]  for o,n in a.items() }
        self.getlastname={n:o for o,n in a.items()}


def make_canvas_and_spacemaps(Y1, Y2, hungmatch, normalize=True, maxerr=0):
    row_ind, col_ind = hungmatch
    pairs = zip(Y1[row_ind], Y2[col_ind])
    pairs = Counter(pairs)  # pair:occurance

    # normalize for old size
    clustersizes = Counter(Y1)  # class:occurence
    clustersizes2 = Counter(Y2)  # class:occurence

    size1 = sum(clustersizes.values())
    size2 = sum(clustersizes2.values())
    if size1 > size2: size1, size2 = size2, size1
    sizemulti = 1  # float(size1)/float(size2)

    y1map = spacemap(clustersizes.keys())
    y2map = spacemap(clustersizes2.keys())

    if normalize:
        normpairs = {k: (2 * sizemulti * float(-v)) / float(clustersizes[k[0]] + clustersizes2[k[1]]) for k, v in
                     pairs.items()}  # pair:relative_occur

    else:
        normpairs = {k: -v for k, v in pairs.items()}  # pair:occur

    canvas = np.zeros((y1map.len, y2map.len), dtype=float)
    for (a, b), v in normpairs.items():
        if v < -maxerr:
            canvas[y1map.getint[a], y2map.getint[b]] = v

    return y1map, y2map, canvas


class spacemap():
    # we need a matrix in with groups of clusters are the indices, -> map to integers
    def __init__(self, items):
        self.itemlist = items
        self.integerlist = list(range(len(items)))
        self.len = len(items)
        self.getitem = { i:k for i,k in enumerate(items)}
        self.getint = { k:i for i,k in enumerate(items)}


def finalrename(Y1, Y2, y1map, y2map, row_ind, col_ind, rn1, rn2):
    tuplemap = [((y1map.getitem[r],), (y2map.getitem[c],), 43) for r, c in zip(row_ind, col_ind)]
    return rename(tuplemap, Y1, Y2, rn1, rn2)


def clean_matrix(canvas):
    canvasbackup = np.array(canvas)
    aa, bb = np.nonzero(canvas)
    for a, b in zip(aa, bb):
        if canvas[a, b] > min(canvas[a, :]) and canvas[a, b] > min(canvas[:, b]):
            canvas[a, b] = 0
            continue
        if canvas[a, b] / np.sum(canvasbackup[a, :]) < .3 and canvas[a, b] / np.sum(canvasbackup[:, b]) < .3:
            canvas[a, b] = 0
    return canvas, canvasbackup


def rename(tuplemap, Y1, Y2, rn1, rn2):
    da = {}
    db = {}
    for i, (aa, bb, _) in enumerate(tuplemap):

        seen = set()
        for a in aa:
            if a in da:
                seen.add(da[a])
        for b in bb:
            if b in db:
                seen.add(db[b])

        if len(seen) == 1:
            target = seen.pop()
        elif len(seen) == 0:
            target = i
        else:
            print("ERROR in rename hungutil")

        for a in aa:
            if a not in da:
                da[a] = target
        for b in bb:
            if b not in db:
                db[b] = target

    lastclasslabel = max(max(da.values()), max(db.values()))
    unmatch = {}

    def unmatched(item, unmatch, lastclasslabel):
        if item not in unmatch:
            unmatch[item] = lastclasslabel + 1
            lastclasslabel += 1
        return unmatch[item], lastclasslabel

    for e in Y1:
        if e not in da:
            da[e], lastclasslabel = unmatched((e, 1), unmatch, lastclasslabel)

    for e in Y2:
        if e not in db:
            db[e], lastclasslabel = unmatched((e, 2), unmatch, lastclasslabel)

            # loggin the renaming
    rn1.total_rename(da)
    rn2.total_rename(db)

    r1 = np.array([da.get(e) for e in Y1])
    r2 = np.array([db.get(e) for e in Y2])
    return r1, r2


def recluster(data, Y, problemclusters, n_clust=2, rnlog=None, debug=False, showset={}):
    # data=umap.UMAP(n_components=2).fit_transform(data)
    indices = [y in problemclusters for y in Y]
    data2 = data[indices]

    yh = predictgmm(n_clust, data2)
    maxy = np.max(Y)
    Y[indices] = yh + maxy + 1

    rnlog.log(problemclusters, np.unique(yh) + maxy + 1)
    if debug or 'renaming' in showset:
        print('ranaming: set%s' % rnlog.dataset, problemclusters, np.unique(yh) + maxy + 1)
    return Y


def split_and_mors(Y1, Y2, hungmatch, data1, data2,
                   debug=False,
                   normalize=True,
                   maxerror=.15,
                   rn=None,
                   saveheatmap=None,
                   showset=None, distmatrix=None):
    '''
    rn is for the renaming log
    '''

    rn1, rn2 = rn
    # get a mapping
    row_ind, col_ind = hungmatch
    y1map, y2map, canvas = make_canvas_and_spacemaps(Y1, Y2, hungmatch, normalize=normalize)
    row_ind, col_ind = solve_dense(canvas)
    canvas, canvasbackup = clean_matrix(canvas)
    if debug or 'inbetweenheatmap' in showset:
        draw.doubleheatmap(canvasbackup, canvas, y1map, y2map, row_ind, col_ind)

    #  da and db are dictionaries pointing out mappings to multiple clusters in the other set
    aa, bb = np.nonzero(canvas)
    da = defaultdict(list)
    db = defaultdict(list)
    for a, b in zip(aa, bb):
        da[a].append(b)
        db[b].append(a)
    da = {a: b for a, b in da.items() if len(b) > 1}
    db = {a: b for a, b in db.items() if len(b) > 1}
    done = True

    for a, bb in da.items():
        if any([b in db for b in bb]):  # do nothing if the target is conflicted in a and b
            continue
        recluster(data1, Y1, [y1map.getitem[a]], n_clust=len(bb), rnlog=rn1, debug=debug, showset=showset)
        # print(f"reclustered {y1map.getitem[a]} of data1 into {len(bb)}")
        done = False

    for b, aa in db.items():
        if any([a in da for a in aa]):  # do nothing if the target is conflicted in a and b
            continue
        recluster(data2, Y2, [y2map.getitem[b]], n_clust=len(aa), rnlog=rn2, debug=debug, showset=showset)
        # print(f"reclustered {y2map.getitem[b]} of data2 into {len(aa)}")
        done = False

    if done:
        row_ind, col_ind = solve_dense(canvas)

        # remove matches that have a zero as value
        row_ind, col_ind = list(zip(*[(r, c) for r, c in zip(row_ind, col_ind) if canvas[r, c] < 0]))
        Y1, Y2 = finalrename(Y1, Y2, y1map, y2map, row_ind, col_ind, rn1, rn2)

        # this is to draw the final heatmap...
        # row_ind, col_ind = hungmatch
        y1map, y2map, canvas = make_canvas_and_spacemaps(Y1, Y2, hungmatch, normalize=normalize)
        # row_ind, col_ind = solve_dense(canvas)
        canvas, canvasbackup = clean_matrix(canvas)
        if debug or 'heatmap' in showset:
            draw.doubleheatmap(canvasbackup, canvas, y1map, y2map, row_ind, col_ind, save=saveheatmap)

        if 'sankey' in showset:
            zzy1map, zzy2map, zzcanvas = make_canvas_and_spacemaps(Y1, Y2, hungmatch, normalize=False)
            print("sankey: raw")
            draw.sankey(zzcanvas, zzy1map, zzy2map)
            print("sankey: normalized")
            draw.sankey(canvasbackup, y1map, y2map)
            print("sankey: cleaned up")
            draw.sankey(canvas, y1map, y2map)

        if "drawdist" in showset:
            draw.distrgrid(distmatrix, Y1, Y2, hungmatch)
        # NOT WE NEED TO PRINT A BEAUTIFUL TABLE

        classes = list(set(np.unique(Y1)).union(set(np.unique(Y2))))
        classes.sort()

        # first 3 columns
        out = [classes]
        out.append([rn1.rn.get(e, -1) for e in classes])
        out.append([rn2.rn.get(e, -1) for e in classes])
        # the next ones are more tricky
        y1map, y2map, canvas = make_canvas_and_spacemaps(Y1, Y2, hungmatch, normalize=False)
        y1cnt = Counter(Y1)
        y2cnt = Counter(Y2)
        out.append([y1cnt.get(e, 0) for e in classes])
        out.append([y2cnt.get(e, 0) for e in classes])
        a = []
        b = []
        for e in classes:
            if e in y1map.getint and e in y2map.getint:
                a.append(
                    "%.2f" % (np.abs(canvas[y1map.getint[e], y2map.getint[e]]) / min(y1cnt.get(e, 0), y2cnt.get(e, 0))))
                b.append(np.abs(canvas[y1map.getint[e], y2map.getint[e]]))
            else:
                a.append(0)
                b.append(0)
        out.append(a)
        out.append(b)

        out = list(zip(*out))

        if debug or 'table' in showset:
            print(tabulate.tabulate(out, ['clusterID', 'ID set 1', 'ID set 2', 'size set 1', 'size set 2', 'matches',
                                          "# matches"]))
        if debug or 'table_latex' in showset:
            print(tabulate.tabulate(out, ['clusterID', 'ID set 1', 'ID set 2', 'size set 1', 'size set 2', 'matches',
                                          "# matches"], tablefmt='latex'))

        # print ("renaming",rn1.rn)
        # print ("renaming",rn2.rn)
        return Y1, Y2, out

    #########################
    if debug: draw.cmp2(Y1, Y2, data1, data2)

    return split_and_mors(Y1, Y2, hungmatch, data1, data2, debug=debug, normalize=normalize, maxerror=maxerror,
                         rn=(rn1, rn2), saveheatmap=saveheatmap, showset=showset, distmatrix=distmatrix)


def bit_by_bit(mata, matb, claa, clab,
               debug=True, normalize=True, maxerror=.13,
             saveheatmap=None, showset={}):
    t = time.time()
    a, b, dist = hungarian(mata, matb, debug=debug)
    hungmatch = (a, b)
    if "time" in showset: print(f'hungmatch took {time.time() - t}')
    # return find_clustermap_one_to_one_and_split(claa,clab,hungmatch,mata,matb,debug=debug,normalize=normalize,maxerror=maxerror)
    # claa,clab =  make_even(claa, clab,hungmatch,mata,matb,normalize)
    rn1 = renamelog(np.unique(claa), '1')
    rn2 = renamelog(np.unique(clab), '2')
    return split_and_mors(claa, clab, hungmatch, mata, matb, debug=debug,
                          normalize=normalize,
                          maxerror=maxerror,
                          rn=(rn1, rn2),
                          saveheatmap=saveheatmap,
                          showset=showset,
                          distmatrix=dist)


import ubergauss as ug
def cluster_ab(a,b):
    d= {"nclust_min":9, "nclust_max":9, "n_init": 10}
    return ug.get_model(a,**d).predict(a), ug.get_model(b,**d).predict(b)
    





##################
#  old stuff below, probably not used anymore
##################




"""




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
    cost_by_clusterindex = lambda x,y : canvas[y1map.getint[x],y2map.getint[y]] # cost accessed by old indices
    subcosts = [ get_min_subcost(y1map.getitem[r],  y2map.getitem[c], cost_by_clusterindex) for r,c in zip(row_ind,col_ind) ]
    
    if debug: # draw heatmap # for a good version of this check out notebook 10.1
        
        if debug:
            debug_canvas = np.zeros( (y1map.len, y2map.len ),dtype=float )
            for clusters_a in y1map.itemlist:
                for clusters_b in y2map.itemlist:
                    debug_canvas[y1map.getint[clusters_a],y2map.getint[clusters_b]] = \
                        -1 * sum([ pairs[c,d] for c in clusters_a for d in clusters_b ]) 
        
        # HEATMAP normalized 
        draw.heatmap(canvas, y1map, y2map, row_ind, col_ind)
        # HEATMAP total 
        draw.heatmap(debug_canvas, y1map, y2map, row_ind, col_ind)
        
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


''' 
def decorate(items):
    l = range(len(items))
    return [' '.join(map(str,items[i]))for i in l ]
''' 



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
    assert duplicates(y1names) , 'old assignments went wrong, call multimapclusters with debug'
    assert duplicates(y2names) , 'old assignments went wrong, call multimapclusters with debug '
    y1names = {i:y1map.getint[a] for a in y1names for i in a }
    y2names = {i:y2map.getint[a] for a in y2names for i in a }
   
    # now we only need to translate
    Yh = ba.lmap(lambda x:y1names.get(x,-2) , Yh) 
    Y2h = ba.lmap(lambda x:y2names.get(x,-2) , Y2h) 

    clustermap = {y1map.getint[a]:y2map.getint[b] for a,b in clustermap }
    class_acc = qualitymeassure( Yh ,Y2h , hungmatch,clustermap)

    return Yh,Y2h, {clustermap.get(a):"%.2f" % b for a,b in class_acc[:-1]},class_acc[-1][1] 



def leftoverassign(canvas, y1map,y2map, row_ind, col_ind):
    # assign leftover classes to the best fit 
    diff_a = list( set(y1map.integerlist) - set(row_ind))
    diff_b = list( set(y2map.integerlist) - set (col_ind))
    log = []
    while len(diff_a) > 0:
        element = diff_a.pop()
        row_ind = np.concatenate((row_ind,[element]))
        best_hit = np.argmin(canvas[element,:])
        col_ind= np.concatenate((col_ind,[best_hit]))
        log.append(('ax0',best_hit))
   
    while len(diff_b) > 0:
        element = diff_b.pop()        
        col_ind = np.concatenate((col_ind,[element]))
        best_hit=np.argmin(canvas[:,element])
        row_ind= np.concatenate((row_ind,[best_hit]))
        log.append(('ax1',best_hit))
        
    return row_ind, col_ind, log
   



# annotate LOSSes for assigning weirdo clusters ,,,, for ROw and COlumn
def rocoloss(a,b,y1map,y2map,canvas):
    a=y1map.getint[a]
    b=y2map.getint[b]
    v = canvas[a,b]
    #amax = np.min(canvas[[x!=a for x in y1map.integerlist],b])#-canvas[a,b]
    amax = np.min(canvas[:,b])-canvas[a,b] # previous solution which is nice
    offender_a = y1map.getitem[np.argmin(canvas[:,b])]
    bmax = np.min(canvas[a,:])-canvas[a,b]
    offender_b = y2map.getitem[np.argmin(canvas[a,:])]
    return (amax,bmax,offender_a,offender_b)
        
def find_clustermap_one_to_one(Y1,Y2, hungmatch, debug=False, normalize=True):
    
    row_ind, col_ind = hungmatch
    y1map,y2map,canvas = make_canvas_and_spacemaps(Y1,Y2,hungmatch,normalize=normalize)
    
    # MAKE ASSIGNMENT 
    row_ind, col_ind = solve_dense(canvas)
        
    # assign leftovers to best hit
    row_ind, col_ind, _ = leftoverassign(canvas, y1map,y2map, row_ind, col_ind)
    
    
    
    
    result =  [ ((a,),(b,),rocoloss(a,b,y1map,y2map,canvas)) for a,b in zip([y1map.getitem[r] for r in row_ind],[y2map.getitem[c] for c in  col_ind])]
    
    if debug: 
        draw.heatmap(canvas, y1map, y2map, row_ind, col_ind)
        print(" clsuter in first set, old in second set,  (loss in row, loss in col, reason for row loss, reason for col loss)")
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
    assert duplicates(y1names) , 'old assignments went wrong, call multimapclusters with debug'
    assert duplicates(y2names) , 'old assignments went wrong, call multimapclusters with debug '
    
    
    y1names = {i:y1map.getint[a] for a in y1names for i in a }
    y2names = {i:y2map.getint[a] for a in y2names for i in a }
   
    # now we only need to translate
    Yh = ba.lmap(lambda x:y1names.get(x,-2) , Yh) 
    Y2h = ba.lmap(lambda x:y2names.get(x,-2) , Y2h) 

    clustermap = {y1map.getint[a]:y2map.getint[b] for a,b in clustermap }
    class_acc = qualitymeassure( Yh ,Y2h , hungmatch,clustermap)

    return Yh,Y2h, {clustermap.get(a):"%.2f" % b for a,b in class_acc[:-1]},class_acc[-1][1] 









##############
# we do 1:1 and do some splitting until all is good 
###############
 
        
    
    
def find_clustermap_one_to_one_and_split(Y1,Y2, hungmatch, data1,data2, debug=False, normalize=True,maxerror=.15):
    
    
    # MAP, ASSIGN THE LEFTOVERS
    row_ind, col_ind = hungmatch
    y1map,y2map,canvas = make_canvas_and_spacemaps(Y1,Y2,hungmatch,normalize=normalize)
    row_ind, col_ind = solve_dense(canvas)
    row_ind, col_ind, log= leftoverassign(canvas, y1map,y2map, row_ind, col_ind)
    
    
    # EXPERIMANTAL INBETWEEN MERGE 
    tupmap =  [ ((a,),(b,),666) for a,b in zip([y1map.getitem[r] for r in row_ind],[y2map.getitem[c] for c in  col_ind])]
    Y1,Y2 = rename(tupmap,Y1,Y2)
    y1map,y2map,canvas = make_canvas_and_spacemaps(Y1,Y2,hungmatch,normalize=normalize)
    row_ind, col_ind = solve_dense(canvas)
    
    
    
    # Mappings in old-name-space with scores
    result =  [ ((a,),(b,),rocoloss(a,b,y1map,y2map,canvas)) for a,b in zip([y1map.getitem[r] for r in row_ind],[y2map.getitem[c] for c in  col_ind])]
    split = [ (c+d,c,e,1) for a,b,(c,d,e,f) in result if  c < -maxerror]+[ (c+d,d,f,2) for a,b,(c,d,e,f) in result if  d < -maxerror]
    
    
    if debug: 
        draw.heatmap(canvas, y1map, y2map, row_ind, col_ind)
        print(" clsuter in first set, old in second set,  (loss in row, loss in col, reason for row loss, reason for col loss)")
        pprint.pprint(result)
        
    # ARE WE FINISHED?
    if not split: 
        return rename(result,Y1,Y2)
    
    # SPLIT PROBLEMATIC CLUSTERS
    split.sort()
    totalcost,cost_in_split_direction,cluster,where = split[0]
    if debug: print (f'splitting old {cluster} of set {where}')
    if where ==1: 
        Y1 = recluster(data1,Y1,[cluster])
    elif where ==2:
        Y2 = recluster(data2,Y2,[cluster])
    
    
    if debug: draw.cmp(Y1, Y2, data1, data2)
    Y1,Y2 = rename(result,Y1,Y2) # this should take care of eccessice splits... 
    return find_clustermap_one_to_one_and_split(Y1,Y2,hungmatch,data1,data2,debug=debug,normalize=normalize)
    
######################
# split and merge
######################
    
def split_and_merge(Y1,Y2, hungmatch, data1,data2, debug=False, normalize=True,maxerror=.15):
    # like 1on1 with split but now we have a better plan:
    # split first, then merge
    
    
    # MAP, including leftovers
    row_ind, col_ind = hungmatch
    y1map,y2map,canvas = make_canvas_and_spacemaps(Y1,Y2,hungmatch,normalize=normalize)
    row_ind, col_ind = solve_dense(canvas)
    
    
    # Mappings in old-name-space with scores
    result =  [ ((a,),(b,),rocoloss(a,b,y1map,y2map,canvas)) for a,b in zip([y1map.getitem[r] for r in row_ind],[y2map.getitem[c] for c in  col_ind])]
    split = [ (c+d,c,e,1) for a,b,(c,d,e,f) in result if  c < -maxerror]+[ (c+d,d,f,2) for a,b,(c,d,e,f) in result if  d < -maxerror]
    
    if debug: 
        draw.heatmap(canvas, y1map, y2map, row_ind, col_ind)
        print(f" clsuter in first set, old in second set,  (loss in row, loss in col, reason for row loss, reason for col loss) {maxerror}")
        pprint.pprint(result)
        
        
    # ARE WE FINISHED?
    if not split: 
        return rename(result,Y1,Y2)
    
    
    # SPLIT PROBLEMATIC CLUSTER
    split.sort()
    totalcost,cost_in_split_direction,cluster,where = split[0]
    if debug: print (f'splitting old {cluster} of set {where}')
    if where ==1: 
        Y1 = recluster(data1,Y1,[cluster])
    elif where ==2:
        Y2 = recluster(data2,Y2,[cluster])
    
    # NOW WE MERGE 
    y1map,y2map,canvas = make_canvas_and_spacemaps(Y1,Y2,hungmatch,normalize=normalize)
    row_ind, col_ind = solve_dense(canvas)
    row_ind, col_ind, log= leftoverassign(canvas, y1map,y2map, row_ind, col_ind)
    tupmap =  [ ((a,),(b,),666) for a,b in zip([y1map.getitem[r] for r in row_ind],[y2map.getitem[c] for c in  col_ind])]
    Y1,Y2 = rename(tupmap,Y1,Y2)    
  
    if debug: draw.cmp(Y1, Y2, data1, data2)
    return split_and_merge(Y1,Y2,hungmatch,data1,data2,debug=debug,normalize=normalize,maxerror=maxerror)

def oneonesplit(mata,matb,claa,clab, debug=True,normalize=True,maxerror=.13):
    hungmatch = hungarian(mata,matb)
    #return find_clustermap_one_to_one_and_split(claa,clab,hungmatch,mata,matb,debug=debug,normalize=normalize,maxerror=maxerror)
    
    claa,clab =  make_even(claa, clab,hungmatch,mata,matb,normalize)
    
    return split_and_merge(claa,clab,hungmatch,mata,matb,debug=debug,normalize=normalize,maxerror=maxerror)




###############
# JUST SPLIT ALL DAY EVERY DAY 
##############
def split_and_split(Y1,Y2, hungmatch, data1,data2, debug=False, normalize=True,maxerror=.15):
    # like 1on1 with split but now we have a better plan:
    # split first, then merge
    
    # MAP, including leftovers
    row_ind, col_ind = hungmatch
    y1map,y2map,canvas = make_canvas_and_spacemaps(Y1,Y2,hungmatch,normalize=normalize)
    row_ind, col_ind = solve_dense(canvas)
    
    # Mappings in old-name-space with scores
    result =  [ ((a,),(b,),rocoloss(a,b,y1map,y2map,canvas)) for a,b in zip([y1map.getitem[r] for r in row_ind],[y2map.getitem[c] for c in  col_ind])]
    split = [ (c+d,c,e,1) for a,b,(c,d,e,f) in result if  c < -maxerror]+[ (c+d,d,f,2) for a,b,(c,d,e,f) in result if  d < -maxerror]
    
    if debug: 
        draw.heatmap(canvas, y1map, y2map, row_ind, col_ind)
        print(f" clsuter in first set, old in second set,  (loss in row, loss in col, reason for row loss, reason for col loss) {maxerror}")
        pprint.pprint(result)
        
        
    # ARE WE FINISHED?
    if not split: 
        return rename(result,Y1,Y2)
    
    
    # SPLIT PROBLEMATIC CLUSTER
    for totalcost,cost_in_split_direction,cluster,where in split:
        if debug: print (f'splitting old {cluster} of set {where}')
        if where ==1: 
            Y1 = recluster(data1,Y1,[cluster])
        elif where ==2:
            Y2 = recluster(data2,Y2,[cluster])
    
    # NOW WE MERGE 
    Y1,Y2 = make_even(Y1, Y2,hungmatch,data1,data2,normalize)
    if debug: draw.cmp(Y1, Y2, data1, data2)
    return split_and_split(Y1,Y2,hungmatch,data1,data2,debug=debug,normalize=normalize,maxerror=maxerror)


def make_even(claa, clab,hungmatch,mata,matb,normalize):
    # make class counts even
    y1map,y2map,canvas = make_canvas_and_spacemaps(claa,clab,hungmatch,normalize=normalize)
    row_ind, col_ind = solve_dense(canvas)
    row_ind, col_ind, log= leftoverassign(canvas, y1map,y2map, row_ind, col_ind)
    tupmap =  [ ((a,),(b,)) for a,b in zip([y1map.getitem[r] for r in row_ind],[y2map.getitem[c] for c in  col_ind])]
    for aa,bb in upwardmerge(tupmap):
        if len(aa)>1:
            clab = recluster(matb,clab,bb,len(aa))
        if len(bb)>1:
            claa = recluster(mata,claa,aa,len(bb))
    return claa, clab 

def oneonesplitsplit(mata,matb,claa,clab, debug=True,normalize=True,maxerror=.13):
    hungmatch = hungarian(mata,matb)
    #return find_clustermap_one_to_one_and_split(claa,clab,hungmatch,mata,matb,debug=debug,normalize=normalize,maxerror=maxerror)
    claa,clab =  make_even(claa, clab,hungmatch,mata,matb,normalize)
    return split_and_split(claa,clab,hungmatch,mata,matb,debug=debug,normalize=normalize,maxerror=maxerror)





###############
# BIT BY BIT
##############
# split and merge or slkit    

def calc_purity(m,err):
        #print(m)
        m[m>-err]=0
        #print(m)
        m=m.tolist()
        m.sort()
        if np.min(m)==0:
            return 0
        #ma=min(m)
        #m=[mm/ma for mm in m]
        return sum(m[1:])/m[0] # m0 is the largest. 

def find_problems(m,r,c,err):
    m=np.array(m)
    m[r,c]=0
    m[m>-err] = 0
    return np.nonzero(m)

def process_problems(p,canvas,r,c):
    problems = list(zip(*p)) 
    ro= dict( zip(r,c))
    co= dict(zip(c,r))
    
    for a,b in problems: 
        
        val = canvas[a,b]
        # vertical 
        valv = canvas[co[b],b] if b in co else 0
        # horizontal 
        valh = canvas[a,ro[a]] if a  in ro else 0
        
        if val > np.min(canvas[a,:]) and val > np.min(canvas[:,b]):
            print ("we should have taken care of this already")
            continue 
        
        if max(valh,val) > max(valv,val): # more in common vertically 
            res = ((a,co[b]),(b,))

        else: # more in common horizontaly 
            res= ((a,),(b,ro[a]))
       
        importance= valv/valh if valv>valh else valh/valv
        #importance = valv+valh+val
        yield importance,  res
"""
