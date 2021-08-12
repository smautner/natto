
# %%
#######################
import numpy as np
import matplotlib.pyplot as plt 
import umap
from natto import input
from natto.out.quality import spacemap
from lmz import *
def get_matrix(median=False, path = "/home/ikea/data/100x.lst"): 
    alldata = eval(open(path,'r').read())
    alldata=np.array(alldata)
    if not median: 
        return alldata
    alldata[alldata==-1]=np.nan
    z = np.nanmean(alldata,axis=2)
    return z 


# load distance matrix for 100x100 and the labels
matrix = get_matrix(median=True)

names = input.get100names("/home/ikea/data/data")



# %%  CLUSTER PERFORMANCE VS SIMILARITY , SIM > .3
###############################
import seaborn as sns

def docalc(combi, combi_r, a, ar, b, br, combi_natto, combi_natto_r):
    # how we calculate the clustering score, there are many options oO 
    #return 2*combi_r - (ar+br)
    #return 2*combi - (a+b)
    return np.mean((ar,br))/combi_r

sametype_cnt = 822 
unrelated_cnt = 10000 - 922 

import pprint
distances=[]
dataset_names = []
STUPID = [] 
for a in range(matrix.shape[0]):
    for b in range(matrix.shape[1]): 
        if a>=b and matrix[a,b]>0.3:
            dataset_names.append((names[a],names[b]))
            distances.append(matrix[a,b].tolist())
            STUPID.append( matrix[a,b])

#cluster_performance = np.array(eval(open("/home/ikea/projects/data/natto/point3.2.ev",'r').read()))
cluster_performance = np.array(eval(open("/home/ikea/projects/data/natto/p3.3.ev",'r').read()))
CUTOFF = .4
dataset_names = [dn for dn,d in zip(dataset_names,STUPID) if d > CUTOFF]
distances = [np.nanmedian(dn) for dn,d in zip(distances,STUPID) if d > CUTOFF]
clusterscores = [np.median(Map(lambda x:docalc(*x),dn)) for dn,d in zip(cluster_performance,STUPID) if d > CUTOFF]


def addbox(rrr, color):
    a,b = Transpose(rrr)
    data = b 
    position = np.median(a)
    plt.boxplot(data,positions=[position])
    data = a 
    position = np.median(b)
    plt.boxplot(data,positions=[position],vert=False)
    #ugdraw.boxplot(b,position)


def draw_cloud(distances, cluster_performance, datalabels):
    same_name=[]
    same_type=[]
    same_nothing=[]
    sns.set_theme(style="darkgrid")


    for dist,cluster,labels in zip(distances, cluster_performance,datalabels):
        n1,n2 = labels
        #if cluster[0][0]>-1:
        simp = dist,cluster
        if n1==n2:
            same_name.append(simp)
        elif n1[:5] == n2[:5]:
            same_type.append(simp)
        else:
            same_nothing.append(simp)

    #addbox(same_name,'red')
    #addbox(same_nothing,'red')
    #addbox(same_type,'red')
    sns.scatterplot(*Transpose(same_nothing),marker='o',color='blue',
        label=f'unrelated ({len(same_nothing)})  {len(same_nothing)/unrelated_cnt:.2f} ')
    sns.scatterplot(*Transpose(same_name),marker='o',color='red', 
        label=f'same dataset({len(same_name)})  {len(same_name)/100:.2f} ')
    sns.scatterplot(*Transpose(same_type),marker='o',color='magenta',
        label=f'same celltype ({len(same_type)})  {len(same_type)/sametype_cnt:.2f}')
    plt.legend()
    
    #x = [x/10 for x in range(3,11)]
    #plt.xticks(x,labels=x)

    """
    allmarks=same_name+same_type+same_nothing
    #print('spearman black',spear(Transpose(s1)))
    s1,s2=Transpose(allmarks)
    mylr = LR() #RANSACRegressor()
    s1=np.array(s1).reshape(-1,1)
    mylr.fit(s1,s2)
    stuff =[.3,1]
    plt.plot(stuff, [mylr.predict([[x]]) for x in stuff])
    """


    plt.xlabel("similarity ")
    plt.ylabel("cluster performance (mean(a,b)/joint)")
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


draw_cloud(distances,clusterscores,dataset_names)

# %%




# TODO SET A CUTOFF 
 
scd = eval(open("/home/ikea/projects/data/natto/smallclu.ev",'r').read()) # small cluster data



CUTOFF = .7
scd = [dn for dn,d in zip(scd,STUPID) if d > CUTOFF]


combiloss = []
singleloss = []
data=[]
for r in scd: 
    re = r[1][0]

    print(re)

    # re has 4 elements: set1+gmm, set1+tunne, set2+gmm, set2+tunnel
    if re != -1:
        tunnel = False
        alldata = re[tunnel]+re[tunnel+2]
        # alldata is now a list of: 1. clustersize, combilos, singleloss
        data+=alldata



# %%

data = np.array(data)
plt.scatter(data[:,0], data[:,2], label = 'single', s=2)
plt.scatter(data[:,0], data[:,1], label = 'combi', s=2)
plt.legend()
plt.show()
# %%
 




scd = eval(open("/home/ikea/projects/data/natto/p3.5.ev",'r').read()) # small cluster data
CUTOFF = .7
scd = [dn for dn,d in zip(scd,STUPID) if d > CUTOFF and len(dn)>1]



#%%
import numpy as np
from lmz import *
data = np.vstack([np.array(Transpose(x)) for x in scd])


# %%

plt.scatter(data[:,0], data[:,1], c=data[:,2], s=2)
plt.legend()
plt.show()
# %%
