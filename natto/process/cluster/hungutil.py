from lmz import *
import time
import tabulate
from collections import Counter
import numpy as np
from collections import  defaultdict
from lapsolver import solve_dense

from natto.process.util import hungarian
from natto.out import draw
import natto.process.cluster.copkmeans as CKM
from sklearn.neighbors import KNeighborsClassifier as KNC



#####
# first some utils
######
from natto.process.cluster import gmm_1
from natto.process.util import spacemap


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

def distances_nrm(dists): 
    V = np.var(dists/np.mean(dists))
    return 1-V

def make_canvas_and_spacemaps(Y1, Y2, hungmatch, normalize=True, maxerr=0, dist = False):
    row_ind, col_ind = hungmatch
    pairs = zip(Y1[row_ind], Y2[col_ind])
    pairs = Counter(pairs)  # pair:occurance

    # normalize for old size
    clustersizes = Counter(Y1)  # class:occurence
    clustersizes2 = Counter(Y2)  # class:occurence

   
    y1map = spacemap(clustersizes.keys())
    y2map = spacemap(clustersizes2.keys())

    if normalize and (dist is not False):
        # see comments below on how this works out
        # new dist thing: pair: distances
        d = defaultdict(list) 
        for a,b,c in zip(Y1[row_ind], Y2[col_ind], dist):
            d[(a,b)].append(c)

        normpairs = {k: (2  * float(-v) * distances_nrm(d[k]) ) / float(clustersizes[k[0]] + clustersizes2[k[1]]) for k, v in
                     pairs.items()}  # pair:relative_occur
        

    elif normalize:
        normpairs = {k: (2  * float(-v)) / float(clustersizes[k[0]] + clustersizes2[k[1]]) for k, v in
                     pairs.items()}  # pair:relative_occur

    else:
        normpairs = {k: -v for k, v in pairs.items()}  # pair:occur

    canvas = np.zeros((y1map.len, y2map.len), dtype=float)
    for (a, b), v in normpairs.items():
        if v < -maxerr:
            canvas[y1map.getint[a], y2map.getint[b]] = v
    return y1map, y2map, canvas


def finalrename(Y1, Y2, y1map, y2map, row_ind, col_ind, rn1, rn2):
    tuplemap = [((y1map.getitem[r],), (y2map.getitem[c],), 43) for r, c in zip(row_ind, col_ind)]
    return rename(tuplemap, Y1, Y2, rn1, rn2)


def clean_matrix(canvas, threshold=0.3):
    canvasbackup = np.array(canvas)
    aa, bb = np.nonzero(canvas)
    for a, b in zip(aa, bb):
        if canvas[a, b] > min(canvas[a, :]) and canvas[a, b] > min(canvas[:, b]):
            canvas[a, b] = 0
            continue
        if canvas[a, b] / np.sum(canvasbackup[a, :]) < threshold and canvas[a, b] / np.sum(canvasbackup[:, b]) < threshold:
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
    yh = gmm_1(data2, nc=n_clust, cov ='tied')
    
    maxy = np.max(Y)
    Y[indices] = yh + maxy + 1

    rnlog.log(problemclusters, np.unique(yh) + maxy + 1)
    if debug or 'renaming' in showset:
        print('ranaming: set%s' % rnlog.dataset, problemclusters, np.unique(yh) + maxy + 1)
    return Y

def recluster_hungmatch_aware(data, Y, problemclusters, n_clust=2, rnlog=None, debug=False, showset={},
                roind=None,coind=None, target_cls=None, Y2=None, algo='knn'):
    
    indices = [y in problemclusters for y in Y]
    data2 = data[indices]
    
    
    #  OLD CRAB 
    # yh = predictgmm(n_clust, data2)
    
    # Nu start 
    roco = dict(zip(roind,coind))
    indices_int_y1 = np.nonzero(indices)[0]
    indices_int_y2 = np.array([roco.get(a,-1) for a in indices_int_y1])
    target_ok_mask = [(Y2[i] in target_cls) if i >=0 else False for i in indices_int_y2]
    indices_int_y1 = indices_int_y1[target_ok_mask]
    indices_int_y2 = indices_int_y2[target_ok_mask]
    if algo == 'copkmeans':
        grps = [ indices_int_y1[ [Y2[i]==targetcls for i in indices_int_y2 ]]  for targetcls in target_cls  ]
        mustlink= [ (a,b) for grp in grps for a in grp for b in grp ] # i hope this produces all the contraints 
        yh = CKM.cop_kmeans(data2,ml=mustlink,k=n_clust)[0]
    else: 
        model=KNC(weights='distance')# might also try uniform 
        train_y = Y2[indices_int_y2] 
        s= spacemap(np.unique(train_y))
        model.fit(data[indices_int_y1], [s.getint[y] for y in train_y])
        yh = model.predict(data2)
    
    ##### Nu end
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
                   showset=[], 
                   distmatrix=None,do_splits=True):
    '''
    rn is for the renaming log
    '''

    rn1, rn2 = rn
    # get a mapping
    #row_ind, col_ind = hungmatch
    CANVASDISTARG = False # False/ distmatrix # this works but just end up 'marking' spread out clusters
    y1map, y2map, canvas = make_canvas_and_spacemaps(Y1, Y2, hungmatch, normalize=normalize, dist = CANVASDISTARG)
    row_ind, col_ind = solve_dense(canvas)
    canvas, canvasbackup = clean_matrix(canvas)
    if debug or 'inbetweenheatmap' in showset:
        draw.doubleheatmap(canvasbackup, canvas, y1map, y2map, row_ind, col_ind)
    done = True


    
    if do_splits:
        #  da and db are dictionaries pointing out mappings to multiple clusters in the other set
        aa, bb = np.nonzero(canvas)
        da = defaultdict(list)
        db = defaultdict(list)
        for a, b in zip(aa, bb):
            da[a].append(b)
            db[b].append(a)
        da = {a: b for a, b in da.items() if len(b) > 1}
        db = {a: b for a, b in db.items() if len(b) > 1}

        for a, bb in da.items():
            if any([b in db for b in bb]):  # do nothing if the target is conflicted in a and b
                continue

            


            # normal recluster
            #recluster(data1, Y1, [y1map.getitem[a]], n_clust=len(bb), rnlog=rn1, debug=debug, showset=showset)
            # reclustering with copkmeanz
            target_classes = [y2map.getitem[b] for b in bb]
            recluster_hungmatch_aware(data1,
                                Y1,
                                [y1map.getitem[a]],
                                n_clust=len(bb),
                                rnlog=rn1,
                                debug=debug,
                                showset=showset,
                                roind=hungmatch[0],
                                coind=hungmatch[1],
                                target_cls= target_classes,
                                Y2=Y2)
            
            done = False

        for b, aa in db.items():
            if any([a in da for a in aa]):  # do nothing if the target is conflicted in a and b
                continue
                
                
                
            #recluster(data2, Y2, [y2map.getitem[b]], n_clust=len(aa), rnlog=rn2, debug=debug, showset=showset)
            # reclustering with copkmeanz
            target_classes = [y1map.getitem[a] for a in aa]
            recluster_hungmatch_aware(data2, Y2, [y2map.getitem[b]], n_clust=len(aa), rnlog=rn2, debug=debug, showset=showset,
                            roind=hungmatch[1],
                            coind=hungmatch[0],
                            target_cls=target_classes,
                            Y2=Y1)
            
            done = False

    # SPECIAL TRIANGLE TREATMENT 
    if False:
        for a, bb in da.items():
            print ("scanning a line for triangles.. ",y1map.getitem[a],Map(y2map.getitem.get,bb), end='')
            if any([b in db for b in bb]): 
                #assert len(bb)==2,f"attempt to solve triangle encountered strange circumstances {bb}"
                # killvectors
                k1a, k1b_ = a,bb
                
                for b in bb: 
                    if b in db:
                        k2b, k2a_ = b,db[b]
                        break
                killa, killb  = min( [(k1a, k1b_[0]),
                                         (k1a, k1b_[1]),
                                         (k2a_[0], k2b),
                                         (k2a_[1], k2b)] , key = lambda x: canvas[x])

                # print(f"killab {y1map.getitem[killa]} {y2map.getitem[killb]}")
                # if the vectos intersect the killpoint, execute order 66 
                # ok now we think about how to actually do it 

                Y1copy = Y1.copy()
                if (killa, killb) in [(k1a, k1b) for k1b in k1b_ ]:

                    recluster_hungmatch_aware(data1,
                            Y1, [y1map.getitem[k1a]],
                            n_clust=len(k1b_),
                            rnlog=rn1,
                            debug=debug,
                            showset=showset,
                            roind=hungmatch[0],
                            coind=hungmatch[1],
                            target_cls=[y2map.getitem[asd] for asd in k1b_],
                            Y2=Y2)                   
                    # THIS WILL UPDATE Y1!  -> use the copy below .. 

                if (killa, killb) in [(k2a, k2b) for k2a in k2a_ ]:
                    recluster_hungmatch_aware(data2,
                            Y2, [y2map.getitem[k2b]],
                            n_clust=len(k2a_),
                            rnlog=rn2,
                            debug=debug,
                            showset=showset,
                            roind=hungmatch[1],
                            coind=hungmatch[0],
                            target_cls=[y1map.getitem[asd] for asd in k2a_],
                            Y2=Y1copy)

                done = False 


    if done:
        row_ind, col_ind = solve_dense(canvas)

        # remove matches that have a zero as value
        row_ind, col_ind = list(zip(*[(r, c) for r, c in zip(row_ind, col_ind) if canvas[r, c] < 0]))
        Y1, Y2 = finalrename(Y1, Y2, y1map, y2map, row_ind, col_ind, rn1, rn2)

        # this is to draw the final heatmap...
        # row_ind, col_ind = hungmatch
        y1map, y2map, canvas = make_canvas_and_spacemaps(Y1, Y2, hungmatch, normalize=normalize, dist = CANVASDISTARG)
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

    return split_and_mors(Y1, Y2, hungmatch, data1, data2, debug=debug, normalize=normalize, maxerror=maxerror,
                         rn=(rn1, rn2), saveheatmap=saveheatmap, 
                         showset=showset, distmatrix=distmatrix, do_splits=do_splits)


def bit_by_bit(mata, matb, claa, clab,
               debug=True, normalize=True, maxerror=.13,
             saveheatmap=None, showset={},do_splits=True):
    t = time.time()
    hungmatch, dist = hungarian(mata, matb, debug=debug)
    #hungmatch = (a, b)
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
                          distmatrix=dist, do_splits = do_splits)

