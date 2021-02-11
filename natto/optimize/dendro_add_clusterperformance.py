#!/home/mautner/.myconda/miniconda3/bin/python
import sys
import basics as ba
import numpy as np
from natto.input import load
from natto.input import preprocessing, pp_many
from sklearn.metrics import pairwise_distances, adjusted_rand_score
import ubergauss as ug 
import gc
import natto
dnames = "human1 human2 human3 human4 smartseq2 celseq2 celseq".split()
debug = False

def get_score(name, name2):
    sampnum = 200 if debug else 1000
    d1 = load.loadgruen_single(f"../data/punk/{name}", subsample=sampnum)
    d2 = load.loadgruen_single(f"../data/punk/{name2}", subsample=sampnum)

    pp = preprocessing.Data().fit(d1,d2,
                    debug_ftsel=False,
                    scale=True, 
                    maxgenes = 800)  
    allteX =  np.vstack(pp.dx)
    labels = natto.process.gmm_1(allteX)
    #labels = ug.get_model(allteX, poolsize=1,nclust_min=7, nclust_max=17).predict(allteX)
    #labels = ug.get_model(allteX, poolsize=1,nclust_min=12, nclust_max=12).predict(allteX)
    real_labels = [i for d in [pp.a, pp.b] for i in d.obs['true'].values]
    print (real_labels, labels)
    rands = adjusted_rand_score(real_labels, labels)
    return rands

import dendro_mk_mtx as dmm 

def get_score_100(name, name2):
    sampnum = 200 if debug else 1000
    d1 = load.load100(name)
    d2 = load.load100(name2)
    pp = preprocessing.Data().fit(d1,d2,
                    debug_ftsel=False,
                    scale=True, 
                    maxgenes = 800)  
    allteX =  np.vstack(pp.dx)
    labels = natto.process.gmm_1(allteX)
    #labels = ug.get_model(allteX, poolsize=1,nclust_min=7, nclust_max=17).predict(allteX)
    #labels = ug.get_model(allteX, poolsize=1,nclust_min=12, nclust_max=12).predict(allteX)
    real_labels = [i for d in [pp.a, pp.b] for i in d.obs['truth'].values]
    print (real_labels, labels)
    rands = adjusted_rand_score(real_labels, labels)
    return rands

punk = False
if __name__ == "__main__":
    task = int(sys.argv[1])
    if punk:
        mtx = [[0.0, 0.7171103003533041, 0.6782515978674881, 0.5711449978177814, 0.7751568580958308, 0.8279503639443323, 0.8073698867678036], 
                [0.7171103003533041, 0.0, 0.6458226579358887, 0.5870605129913772, 0.6700045301355889, 0.7311682549812814, 0.7832925704664048], [0.6782515978674881, 0.6458226579358887, 0.0, 0.5919020370236961, 0.7309107461572747, 0.8156476626788953, 0.7536937411088679], [0.5711449978177814, 0.5870605129913772, 0.5919020370236961, 0.0, 0.7017388744339382, 0.7656206989999302, 0.744133557048792], [0.7751568580958308, 0.6700045301355889, 0.7309107461572747, 0.7017388744339382, 0.0, 0.6868481130021408, 0.7669906186334121], [0.8279503639443323, 0.7311682549812814, 0.8156476626788953, 0.7656206989999302, 0.6868481130021408, 0.0, 0.7609737420476113], [0.8073698867678036, 0.7832925704664048, 0.7536937411088679, 0.744133557048792, 0.7669906186334121, 0.7609737420476113, 0.0]]
        mtx = np.array(mtx)
        morder = np.argsort(mtx[task])
        result = []
        self = dnames[morder[0]]
        for did in morder[1:]: 
            other = dnames[did]
            result.append( get_score(self, other))
            gc.collect()
            if debug:
                break
        ba.dumpfile(result,"res/"+sys.argv[1]+"_"+sys.argv[2])
        print("all good")
    else: 
        dnames  = dmm.dnames 
        mtx = [[0.6908272461566701, 0.17760821481247316, 0.5856239190470113, 0.20522295703572607, 0.2380727061971306, 0.1744571518936178, 0.15818011470662208, 0.14788688075750284, 0.15195779222607358], [0.17553023412485272, 0.7942462065855885, 0.15549872846384022, 0.1798922283590384, 0.208208520125358, 0.40177026113031883, 0.418227461693783, 0.4834045769853517, 0.5207559578445297], [0.5875469771054237, 0.1537419393720713, 0.7175527506409204, 0.10385238560008195, 0.3096685741559119, 0.13308583398316928, 0.15746179186664627, 0.13749440534001806, 0.12524203511251866], [0.15585706190907414, 0.1552972145839211, 0.10378457027658369, 0.6955654848662118, 0.22344593803323742, 0.12947525339677407, 0.31519919592472034, 0.1454683347092969, 0.12337142263208098], [0.25986451493898993, 0.24112137530485875, 0.3399435010810133, 0.22189184983898733, 0.7249403897228224, 0.18287699050120781, 0.36069793537963474, 0.25528630592860674, 0.17722567355548546], [0.2093962111482493, 0.4018403799367785, 0.16317545658644278, 0.14522144258891087, 0.15310395020984824, 0.8413948754297165, 0.36376554617130585, 0.286120527612111, 0.3529562642333465], [0.16469215602965506, 0.4042566233131632, 0.1325522860924922, 0.3086532759783118, 0.36599089203249685, 0.3848934160909298, 0.8084319796447434, 0.2593633813216794, 0.3158487427064775], [0.16431210896388024, 0.4444980644773999, 0.12171237934810956, 0.14200259333626855, 0.27238676713535304, 0.32104991199054933, 0.25620624780704837, 0.8307609403840097, 0.4642893883232003], [0.17087834195300175, 0.5432836930278041, 0.17878446391763544, 0.11052898790417073, 0.17940842123377992, 0.4042736198006437, 0.31025677664067697, 0.4995749662271819, 0.8113086248305014]]
        mtx = np.array(mtx)
        morder = np.argsort(-mtx[task])
        print(f"mtx:{mtx[task]}")
        print(f"order:{morder}")
        result = []
        self = dnames[morder[0]]
        for did in morder[1:]: 
            other = dnames[did]
            result.append( get_score_100(self, other))
            gc.collect()
            if debug:
                break
        ba.dumpfile(result,"res/"+sys.argv[1]+"_"+sys.argv[2])
        print("all good")


def res(indices,reps): 
    print (dnames)
    for i in range(indices):
        indexrepeats =  np.array([ba.loadfile(f"res/{i}_{r}") for r in range(reps) ]) 
        print ( indexrepeats.mean(axis=0).tolist())
