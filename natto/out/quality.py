from collections import Counter
import scanpy as sc
from  sklearn.neighbors import KNeighborsClassifier as KNC
import numpy as np
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.metrics import adjusted_rand_score as rand
# QUALITY MEASSUREMENTS
import math
from sklearn.metrics import pairwise_distances
from rari import rari
from natto.process.util import spacemap, hungarian
import pandas as pd

# e^-dd / avg 1nn dist 

def gausssim(a,b, ca, cb): 
    ncb=np.unique(cb)
    res = {}
    for aa in np.unique(ca):
        if aa in ncb:
            if sum(ca==aa) < 5 or sum(cb==aa)<5:
                res[aa]="NA"
                continue
            avg1= nndist(a,ca,cb,c = aa)
            avg2= nndist(b,cb,ca, c=aa)
            d= supernndist(a,b,ca,cb,c=aa)
            avg= (avg1+avg2)/2
            #print (aa,d,avg1,avg2)
            res[aa]= math.exp(-(d*d) / (avg*avg))
    res = {k:v for k,v in res.items() if v!="NA"}
    return res


def nndist(m,cla,clb,c=None): 
            instances = m[cla==c]
            neighs = NN(n_neighbors=2).fit(instances)
            distances, _  = neighs.kneighbors(instances)
            return np.mean(distances[:,1])

def supernndist(a,b,ca,cb,c=None):
            instances = a[ca==c]
            neighs = NN(n_neighbors=1).fit(instances)
            distances, _  = neighs.kneighbors(b[cb==c])
            return np.mean(distances)
        
    
def compare_heatmap(y11,y12,y21,y22,mata,matb):
    
    from lapsolver import solve_dense
    from sklearn.metrics.pairwise import euclidean_distances as ed
    from natto.process.hungutil import make_canvas_and_spacemaps
    from natto.out.draw import quickdoubleheatmap
    distances = ed(mata,matb)
    hungmatch = solve_dense(distances)


    def prephtmap(y1,y2):
        # returns: canvas,y1map2,y2map2,row,col
        a,b,c = make_canvas_and_spacemaps(y1,y2,hungmatch,normalize=False)
        d,e  = solve_dense(c)
        return c,a,b,d,e

    comp1 = prephtmap(y11,y12)
    comp2 = prephtmap(y21,y22)
    
    quickdoubleheatmap(comp1,comp2)

    def calcmissmatches(stuff):
        canv = stuff[0] 
        r,c = stuff[-2:]
        for rr,cc in zip(r,c):
            canv[rr,cc]=0
        return canv.sum()

    print("clust1 missplaced:", calcmissmatches(comp1))
    print("clust2 missplaced:", calcmissmatches(comp2))
    print("set1 randindex:",rand(y11,y21) )
    print("set2 randindex:",rand(y12,y22) )







# Clustering  -> use 1NN purity :) 

def clust(nparray, labels):
    neighs = KNC(n_neighbors=2)
    neighs.fit(nparray,labels)
    _, pairs = neighs.kneighbors(nparray)# get neighbor
    acc= sum([labels[a]==labels[b] for a,b in pairs])/len(labels)
    return acc  


def doubleclust(X,X2, Y,Y2):
    def asd(X,X2,Y,Y2):
        neighs = KNC(n_neighbors=1)
        neighs.fit(X,Y)
        _, pairs = neighs.kneighbors(X2)
        return sum([ Y[a]==b  for a,b in zip(pairs,Y2)])/len(Y2)
    
    return (asd(X,X2,Y,Y2)+asd(X2,X,Y2,Y))/2



def make_rari_compatible(ar): 
    ''' not matching clusternames cause holes in clusternames causing rari 2 die'''
    s= spacemap(np.unique(ar) )
    ar = np.array(ar)
    for k,v in s.getint.items(): 
        if k!=v:
            ar[ar==k] = v
    return ar
            


def rari_score(Y1,Y2,X1,X2,metric='euclidean'): 
    # Y1 and Y2 are clculated on different data... 
    # account for that by ordering Y2 ; such that the closest cells correspond
    (a, b), dist = hungarian(X1, X2, debug=False, metric = metric)
    aTb = dict(zip(a,b))
    k=list(aTb.keys())
    k.sort()
    new_order = [aTb[kk] for kk in k]
    X2 = X2[new_order]
    Y2 = Y2[new_order]

    return rari_srt(Y1,Y2,X1,X2) 



def  rari_srt(Y1,Y2,X1,X2):
    dist_x1 = pairwise_distances(X1) # clustering happened in euclidean space?
    dist_x2 = pairwise_distances(X2)
    Y1 = make_rari_compatible(Y1)
    Y2 = make_rari_compatible(Y2)
    return rari(Y1,Y2,dist_x1,dist_x2)  



def rand_score(Y1,Y2):
     return rand(Y1,Y2)

def printRARI(y1,y2,x1,x2):
    rarr, rand = rari_score(y1,y2,x1,x2) 
    print(f"RARI {rarr}  ARI {rand}")
    return rarr, rand

from lmz import Map, Zip, Range
    

def make_res_dict( adata): 
    cats = adata.obs['clust'].cat.categories
    label_markers = {}
    for e in cats:
        cat = str(e)
        genes = adata.uns['rank_genes_groups']['names'][cat]
        scores = adata.uns['rank_genes_groups']['scores'][cat]
        ranks = Range(scores)
        label_markers[e] = Zip(genes,scores, ranks)
    return label_markers
    
def print_scorediff(label, d1,d2,num):
    #gen_score_1 = {a:b for a,b,c in d1.get(label,[])}
    gen_score_2 = {a:b for a,b,c in d2.get(label,[])}
    res= []
    minscore = min(gen_score_2.values() or [0] )
    for gene, score, _ in d1.get(label,[]): 
        other_score = gen_score_2.get(gene,minscore)
        res.append( (abs(score-other_score),gene,score-other_score) )
    res.sort()

    def mk_print(r):
        return f"{r[1]}({r[2]:.2f})"
    print (label,"\t",'\t'.join(map(mk_print,res[::-1][:num])))
    
    
    

def markers(m,y1,y2, num=7, diff = True): 
    # add y1 and y2 to data in m 
    cat1 = pd.Categorical(y1)
    cat2 = pd.Categorical(y2)
    m.a.obs['clust'] =  cat1 
    m.b.obs['clust'] =  cat2

    # do ttest analysis 
    sc.tl.rank_genes_groups(m.a, 'clust', method='t-test')
    sc.tl.rank_genes_groups(m.b, 'clust', method='t-test')

    # ok first we get a good datastructure for the data
    d1 = make_res_dict( m.a)
    d2 = make_res_dict( m.b)

    # report stuff 
    def mk_printable(gene_score_rnk, d2_grank):
        gene, score, rnk = gene_score_rnk
        return f'{gene}({d2_grank.get(gene,"n/a")})'
        #return f'{gene}({score:.1f})'

    def print_ranked_genes(k,d,d2):
        d2gen_rnk = {a:c for a,b,c in d2.get(k,[])}
        if k in d:
            print (k,"\t",'\t'.join(map(lambda x:mk_printable(x,d2gen_rnk),d[k][:num])))

    for label in set(list(d1.keys())+list(d2.keys())):
        print_ranked_genes(label,d1,d2)
        print_ranked_genes(label,d2,d1)
        if diff:
            print_scorediff(label, d1,d2, num)
            print_scorediff(label, d2,d1, num)

    # use distance as meassure of criticallity maybe 
    return d1,d2
    

def mk_label_avghungdist(dx, y1,y2):
    (a,b), dist = hungarian(*dx)
    y1dist = dist[a,b]
    #y1dist -= y1dist.min()
    #y1dist /= y1dist.max()

    y2dist = y1dist[np.argsort(b)]

    def mk_label_avgdist(y,dist): 
        return {u: "%.2f" % dist[y==u].mean() for u in np.unique(y)}

    return Map(mk_label_avgdist, [y1,y2], [y1dist, y2dist])

