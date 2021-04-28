
# %%
#######################
import numpy as np
import matplotlib.pyplot as plt 
import umap
import agglo100 as a 


# load distance matrix for 100x100 and the labels
matrix = a.get_matrix(median=True)




# %%  CLUSTER PERFORMANCE VS SIMILARITY , SIM > .3
###############################

def docalc(combi, combi_r, a, ar, b, br, combi_natto, combi_natto_r):
    # how we calculate the clustering score, there are many options oO 
    #return 2*combi_r - (ar+br)
    #return 2*combi - (a+b)
    return np.mean((ar,br))/combi_r

sametype_cnt = 822 
unrelated_cnt = 10000 - 922 

print(same, overlap,unrelated)
import pprint
distances=[]
dataset_names = []
STUPID = [] 
z = np.nanmean(alldata,axis=2) # copy pasted this here just to make sure
for a in range(z.shape[0]):
    for b in range(z.shape[1]): 
        if a>=b and z[a,b]>0.3:
            dataset_names.append((names[a],names[b]))
            distances.append(alldata[a,b].tolist())
            STUPID.append( z[a,b])

cluster_performance = np.array(eval(open("/home/ikea/projects/data/natto/point3.2.ev",'r').read()))
CUTOFF = .3
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
