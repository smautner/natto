'''
Metric? improve  rari on 3k, 

HWAT? CLUSTERING!:  pca to X and then gmm directly?  cov-> full/not? 

WHAT? preproc params.. 
'''

import numpy as np



from clust_stabi import get_data_mp

import basics as ba 
def rep_mp(repeats, *x):
        r = ba.mpmap(get_data_mp,[x]*repeats, chunksize = 1, poolsize = 10)
        print (np.array(r).mean(axis=0))
        print (np.array(r).var(axis=0))

from basics.sgexec import sgeexecuter
def rep(repeats, *x):
        res = []
        for i in range(repeats):
            d = getdata(*x)
            res.append(clustandscore(d))
        print (np.array(res).mean(axis=0))


#############
# maxgenes and mindisp 
########


s=sgeexecuter()
p=[]

def rep_sge(repeats, *x): 
    s.add_job(get_data_mp, [x]*repeats)
    p.append(x)

repf = rep_sge

def calc_print():
    for r,pp in zip(s.execute(), p):
        print (pp)
        print ("mean",np.array(r).mean(axis=0))
        print ('var', np.array(r).var(axis=0))


def optimize_genesel(): 
    pca = 20
    udim = 10
    for mindisp in [ 1.2, 1.25, 1.3, 1.35,1.4,1.45 ]: # mindisp
        repf(200, mindisp, False, pca,udim)
    for maxgenes in [650,700,750,800, 850, 900, 950]: # MAxgenes        
        repf(200,False,maxgenes,pca,udim)


def optimize_PCA():
    mindisp = False
    maxgenes = 900
    #for pca in [0,15,20,25,30, 40 ]:
    for pca in [20,25,30]:
            repf(200,False,maxgenes,pca,500)

def optimize_pcaumap2():
    mindisp = False
    maxgenes = 900
    for pca in [0,15,20,25,30,40]:
        for udim in [8,10,12,14]: # UMAP
                repf(200,False,maxgenes,pca,udim)


#optimize_genesel()
optimize_PCA()
#optimize_pcaumap2()

calc_print()
'''
1. PCA ONLY (all of theese suck) 
(False, 900, 8, 100)
mean [0.79231839 0.696586  ]
var [0.00738164 0.00563242]
(False, 900, 10, 100)
mean [0.83610982 0.69472685]
var [0.0067263  0.00423844]
(False, 900, 12, 100)
mean [0.8338738  0.69270631]
var [0.00827788 0.00459219]
(False, 900, 14, 100)
mean [0.82499034 0.68471529]
var [0.00759942 0.00397252]


2. UMAP ONLY  10 -> 85
(False, 900, 0, 8)
mean [0.83839778 0.80401424]
var [0.00378707 0.00332683]
(False, 900, 0, 10)
mean [0.85170303 0.81110017]
var [0.00263099 0.00287453]
(False, 900, 0, 12)
mean [0.84981655 0.81312223]
var [0.00248209 0.00325524]
(False, 900, 0, 14)
mean [0.84191786 0.81157969]
var [0.00301368 0.00316814]




(False, 900, 15, 8)
mean [0.84743421 0.83623716]
var [0.0039889  0.00288471]
(False, 900, 15, 10)
mean [0.85132402 0.8287274 ]
var [0.00338706 0.00286303]
(False, 900, 15, 12)
mean [0.839118   0.82802169]
var [0.00413398 0.00298589]
(False, 900, 15, 14)
mean [0.85013039 0.82680395]
var [0.00367486 0.00328263]




(False, 900, 20, 8)
mean [0.85815108 0.83126957]
var [0.00382996 0.00259675]
(False, 900, 20, 10)
mean [0.86829138 0.84314381]
var [0.00307012 0.0031001 ]
(False, 900, 20, 12)
mean [0.85833059 0.83733399]
var [0.00346111 0.00279076]
(False, 900, 20, 14)
mean [0.84877089 0.83253756]
var [0.0035715  0.00309558]



(False, 900, 25, 8)
mean [0.86531505 0.83726063]
var [0.00312651 0.00268755]
(False, 900, 25, 10)
mean [0.87476602 0.84801235]
var [0.00284868 0.00312248]
(False, 900, 25, 12)
mean [0.86307009 0.84482252]
var [0.00313189 0.00267558]
(False, 900, 25, 14)
mean [0.86636278 0.84991058]
var [0.00298229 0.0026464 ]

(False, 900, 30, 8)
mean [0.87070701 0.84171533]
var [0.00269863 0.00264527]
(False, 900, 30, 10)
mean [0.86589268 0.83614984]
var [0.0033741  0.00296594]
(False, 900, 30, 12)
mean [0.86326196 0.84303779]
var [0.00340167 0.00308811]
(False, 900, 30, 14)
mean [0.86521281 0.84982783]
var [0.00341402 0.00269911]



(False, 900, 40, 8)
mean [0.85860456 0.83896781]
var [0.00338631 0.00317401]
(False, 900, 40, 10)
mean [0.86823823 0.83948743]
var [0.00340891 0.00272061]
(False, 900, 40, 12)
mean [0.85532975 0.82786205]
var [0.00332962 0.00292057]
(False, 900, 40, 14)
mean [0.8554884  0.84036312]
var [0.00403488 0.00248539]



### THIS TIME I COPY 3K


(False, 900, 8, 100)
mean [0.86089907 0.74020629]
var [0.0100582  0.00706921]
(False, 900, 10, 100)
mean [0.83872643 0.74228459]
var [0.00866715 0.00620865]
(False, 900, 12, 100)
mean [0.83910079 0.73775992]
var [0.0083346  0.00543387]
(False, 900, 14, 100)
mean [0.82893479 0.72745403]
var [0.00800483 0.00613793]



(False, 900, 0, 8)
mean [0.87314862 0.83804043]
var [0.00567105 0.00509965]
(False, 900, 0, 10)
mean [0.86455507 0.84926056]
var [0.006152   0.00562378]
(False, 900, 0, 12)
mean [0.87147699 0.84771134]
var [0.00579631 0.00540775]
(False, 900, 0, 14)
mean [0.87942621 0.84436333]
var [0.00424252 0.00493946]


(False, 900, 15, 8)
mean [0.9192745  0.87133087]
var [0.00392995 0.00594957]
(False, 900, 15, 10)              
mean [0.91824929 0.87893601]
var [0.00411992 0.00568047]
(False, 900, 15, 12)
mean [0.90611389 0.86841195]
var [0.00447011 0.00515283]
(False, 900, 15, 14)
mean [0.89344178 0.8679653 ]
var [0.00585625 0.00558568]

(False, 900, 20, 8)
mean [0.92731257 0.87125755]
var [0.00277729 0.00594225]
(False, 900, 20, 10)
mean [0.91409911 0.8671509 ]
var [0.00373806 0.00589365]
(False, 900, 20, 12)
mean [0.91398862 0.87713727]
var [0.00454915 0.00509655]
(False, 900, 20, 14)
mean [0.90572082 0.8649757 ]
var [0.00404783 0.00595136]


(False, 900, 25, 8)
mean [0.92439374 0.86558673]
var [0.00335349 0.00580994]
(False, 900, 25, 10)
mean [0.91709675 0.88039547]
var [0.00399789 0.00562918]
(False, 900, 25, 12)
mean [0.91953984 0.88359919]
var [0.00354106 0.00484796]
(False, 900, 25, 14)
mean [0.91105525 0.87157405]
var [0.00385463 0.00618352]


(False, 900, 30, 8)
mean [0.92452995 0.85822906]
var [0.00301159 0.00536918]
(False, 900, 30, 10)
mean [0.91639984 0.87409845]
var [0.00400996 0.0057132 ]
(False, 900, 30, 12)
mean [0.91612792 0.87954507]
var [0.00367401 0.00529746]
(False, 900, 30, 14)
mean [0.91476117 0.8711539 ]
var [0.00339164 0.0055065 ]

(False, 900, 40, 8)
mean [0.92164544 0.84984847]
var [0.00286178 0.00563633]
(False, 900, 40, 10)
mean [0.90756704 0.86649947]
var [0.00498605 0.00526088]
(False, 900, 40, 12)
mean [0.91360093 0.86522589]
var [0.00389229 0.00600437]
(False, 900, 40, 14)
mean [0.91107954 0.87219353]
var [0.0036744  0.00527622]

'''
