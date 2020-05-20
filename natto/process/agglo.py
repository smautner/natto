


from sklearn.cluster import AgglomerativeClustering as Agg
from collections import defaultdict

import numpy as np

def precompute_leaves(chilist,lendat):
    dic = defaultdict(list)
    for i, children in enumerate(chilist):
        j=i+lendat
        for child in children:
            if child < lendat:
                dic[j].append(child)
            else:
                dic[j]+=dic[child]
    return dict(dic)
    
    

def cluster(data, data2, matches, link='ward'):
    
    model1 = Agg(distance_threshold =0.00001,n_clusters=None, linkage=link)
    model2 = Agg(distance_threshold =0.00001,n_clusters=None, linkage=link)
    model1.fit(data)
    model2.fit(data2)
    
    l1=precompute_leaves(model1.children_,len(data))
    l2=precompute_leaves(model2.children_,len(data2))
    matchor = dict(zip(*matches))
 
    def correspondence(a,b, return_bool=False):
        id1 = {matchor[z] for z in l1.get(a,[a]) if z in matchor}
        id2 = {z for z in l2.get(b,[b])} 
        inter = len(id1.intersection(id2))
        if return_bool:
            return len(id1)/2 < inter and len(id2)/2 < inter
        return inter
    
    res=[]
    le1,le2 = len(data) , len(data2)
    c1,c2 = model1.children_, model2.children_

    
    ####
    # assign clusters.. 
    #########
    def matchnode(r1,r2): 
        # we know that r1 and r2 match... 
        # -> case 1: children dont match -> terminate (add to results)
        # -> case 2: children match -> they sort it out themselfes
        
        
        # get the child-nodes
        r11,r12 = c1[r1-le1]
        r21,r22 = c2[r2-le2]
        if r11 not in l1 or r12 not in l1 or r21 not in l2 or r22 not in l2:
            # cluster size  of children would be 1: stop
            res.append((r1,r2))
            return
        #if not ( (matchnode(r11,r21) or matchnode (r11,r22)) and (matchnode(r12,r21) or matchnode(r12,r22))):
        #    res.append((r1,r2)) # if the childeren r not matching: we use thos
        
        # how many 
        lenn= (len(l1[r1]) + len(l2[r2]))/2
        align = correspondence(r11,r21, return_bool=False) + correspondence(r12,r22, return_bool=False)
        cross = correspondence(r11,r22, return_bool=False) + correspondence(r12,r21, return_bool=False)
        align/=lenn
        cross/=lenn
        if max(align,cross) < .7:
            # case 1
            res.append((r1,r2))
        else: 
            # case 2
            if align > cross:
                matchnode(r11,r21)
                matchnode(r12,r22)
            else:
                matchnode(r11,r22)
                matchnode(r12,r21)
        return 
                
            
        

    matchnode( max(l1.keys()), max(l2.keys()) )
    res1 = np.full(le1,-1)
    res2 = np.full(le2,-1)
    for i,(a,b) in enumerate(res):
        res1[l1.get(a,[a])]=i
        res2[l2.get(b,[b])]=i
        
    ##########
    # COLORIZING
    ###############
    colormatch = []
    def maxcolor(r1,r2): 
        # stop when we are at a leaf
        if r1 < le1 or r2 < le2:
            #print("maxcolappend",r1,l1.get(r1,None),l1.get(r1,[r1]))
            colormatch.append(  (l1.get(r1,r1),l2.get(r2,r2) ) )
            return 
        
        
        # children_combos:
        r11,r12 = c1[r1-le1] 
        r21,r22 = c2[r2-le2]
        # how to align the branches?
        align = correspondence(r11,r21, return_bool=False) + correspondence(r12,r22, return_bool=False)
        cross = correspondence(r11,r22, return_bool=False) + correspondence(r12,r21, return_bool=False)
        
        left  = (r11,r21) if align > cross else (r11,r22)
        right  = (r12,r22) if align > cross else (r12,r21)
        
        maxcolor(*left)
        maxcolor(*right)
        
    
    maxcolor( max(l1.keys()), max(l2.keys()) )
    

    col1 = np.full(le1,-1)
    col2 = np.full(le2,-1)
    for i,(a,b) in enumerate(colormatch):
        #print(a,b)
        col1[a]=i
        col2[b]=i
    
    return res1, res2, col1,col2
    