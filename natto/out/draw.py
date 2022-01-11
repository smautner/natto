import pandas as pd
from sklearn.cluster import AgglomerativeClustering as agg 
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from umap import UMAP
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

'''
# make a list of colors 
colors = list(permutations([0,.25,.5,.75,1],3))
random.seed(5) #making shuffle consistent
random.shuffle(colors)
f = lambda cm: [cm.colors[i] for i in range(len(cm.colors))]
#colors = f(plt.cm.get_cmap('tab20b')) +f(plt.cm.get_cmap('tab20c')) 
colors = f(plt.cm.get_cmap('tab20'))
colors += f(plt.cm.get_cmap("tab20b"))
colors += f(plt.cm.get_cmap("tab20c"))
# put the list in col for usage
col = { i-2:e for i,e in enumerate(colors)}
col.update( {a+100:b for a,b in col.items()}  )
'''

col = plt.cm.get_cmap('tab20').colors 
col = col+col+col+ ((0,0,0),)

def umap(X,Y,
        title="No title",
        acc : "y:str_description"={}, 
        black = None, 
        show=True,
        markerscale=4,
        getmarker = lambda color: {"marker":'o'},
        size=None):
        
    plt.title(title, size=20)
    Y=np.array(Y)
    size=  max( int(4000/Y.shape[0]), 1) if not size else size
    
    if type(black) != type(None): 
        embed = black
        plt.scatter(embed[:, 0],
                    embed[:, 1],
                    c=[(0,0,0) for e in range(black.shape[0])], 
                    s=size,
                    label='del',marker='X')
    
    embed = X
    for cla in np.unique(Y):
        plt.scatter(embed[Y==cla, 0],
                    embed[Y==cla, 1],
                    color= col[cla],
                    s=size,
                    label= str(cla)+" "+acc.get(cla,''),**getmarker(col[cla]))
    #plt.axis('off')
    plt.xlabel('UMAP 2')
    plt.ylabel('UMAP 1')

    plt.legend(markerscale=markerscale,ncol=5,bbox_to_anchor=(1, -.12) )
    if show: plt.show()
        
        
        
def umap_gradient(X,Y,
        title="No title",
        show=True,
        markerscale=4,
        cmap='gist_rainbow',
        size=None):
    
    # Y is
    plt.title(title, size=20)
    Y=np.array(Y)
    size=  max( int(4000/Y.shape[0]), 1) if not size else size
    
    embed = X
   
    plt.scatter(embed[:, 0],
                embed[:, 1],
                c= Y,
                s=size,cmap=cmap)
    #plt.axis('off')
    plt.xlabel('UMAP 2')
    plt.ylabel('UMAP 1')
    #plt.legend(markerscale=markerscale,ncol=5,bbox_to_anchor=(1, -.12) )
    if show: plt.show()
        
        
        
def cmp2(Y1,Y2,X1,X2,title=('1','2'), save=None, labelappend=[{},{}], noshow=False):

    sns.set(font_scale=1.2,style='white')
    plt.figure(figsize=(16,8))    

    same_limit=True
    if same_limit:
        X  = np.concatenate((X1, X2), axis=0)
        xmin,ymin = X.min(axis = 0) 
        xmax,ymax = X.max(axis = 0) 

    #plt.tight_layout()    
    ax=plt.subplot(121)
    if same_limit:
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
    umap(X1,Y1,show=False,title=title[0],size=False,markerscale=4, acc=labelappend[0])
    ax=plt.subplot(122)
    if same_limit:
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
    umap(X2,Y2,show=False,title=title[1],size=False,markerscale=4 , acc=labelappend[1])
    if save:
        plt.tight_layout()
        plt.savefig(save, dpi=300)
    if not noshow:
        plt.show()



def cmp2_genes(data,X1,X2,gene, cmap = 'viridis'):
    Y1 = data.a[:,data.a.var['gene_ids'].index == gene].X.T.todense()
    Y2 = data.b[:,data.b.var['gene_ids'].index == gene].X.T.todense()
    cmp2_grad(Y1,Y2, X1,X2,[f'{title}[{gene}]' for title in data.titles], cmap=cmap)
    




def cmp2_grad(Y1,Y2,X1,X2,title=('1','2'),
        save=None,
        fix_colors=True,size=4,
        cmap ='autumn',noshow=False):

    sns.set(font_scale=1.2,style='white')
    plt.figure(figsize=(16,8))    
    
    
    #plt.tight_layout()    
    ax=plt.subplot(121)
    umap_gradient(X1,Y1,show=False,title=title[0],size=size,markerscale=4, cmap = cmap)
    ax=plt.subplot(122)
    umap_gradient(X2,Y2,show=False,title=title[1],size=size,markerscale=4, cmap = cmap)
    if save:
        plt.tight_layout()
        plt.savefig(save, dpi=300)
    plt.colorbar()
    if not noshow:
        plt.show()
    
def plot_blobclust(Y1,X1,X2,red=None, save=None):    
    sns.set(font_scale=1.2,style='white')
    if not red:
        red = UMAP()
        red.fit(np.vstack((X1,X2)))
    plt.figure(figsize=(12,12))    
    #plt.tight_layout()     old markers.. 
    #umap(X1,Y1[:X1.shape[0]],red,show=False,title="combined clustering",size=30,markerscale=4,marker='_')
    #umap(X2,Y1[X1.shape[0]:],red,show=False,title="combined clustering",size=30,markerscale=4,marker='|') 
    fill = lambda col: {"marker":'o'}
    empty = lambda col: {'facecolors':'none', 'edgecolors':col  }
    #fill = lambda col: {"marker": "o"}
    #empty = lambda col:{"marker": mpl.markers.MarkerStyle('o','none')}  #{"marker":'o','fillstyle':'none'}


    umap(X1,Y1[:X1.shape[0]],red,show=False,title="combined clustering",size=30,markerscale=4,getmarker= fill)
    umap(X2,Y1[X1.shape[0]:],red,show=False,title="combined clustering",size=30,markerscale=4,getmarker=empty) 
    if save:
        plt.tight_layout()
        plt.savefig(save, dpi=300)
    plt.show()

def cmp3(Y1,Y2,X1,X2,title=('1','2'),red=None, save=None):    
    '''add a comparison, where all labels are kept'''

    sns.set(font_scale=1.2,style='white')
    if not red:
        red = UMAP()
        red.fit(np.vstack((X1,X2)))
    plt.figure(figsize=(24,8))    

    #plt.tight_layout()    
    ax=plt.subplot(131)
    umap(X1,Y1,red,show=False,title=title[0],size=4,markerscale=4)
    ax=plt.subplot(132)
    umap(X2,Y2,red,show=False,title=title[1],size=4,markerscale=4)
    ax=plt.subplot(133)
    umap(np.vstack((X1,X2)),
            [1]*len(Y1)+[2]*len(Y2),
            red,
            show=False,
            title="Combined",
            size=4,
            markerscale=4, acc={1:title[0] , 2:title[1]})
    if save:
        plt.tight_layout()
        plt.savefig(save,dpi=300)
    plt.show()

def venn(one,two: 'boolean array', labels :"string tupple"):
    import  matplotlib_venn as vvv
    selover = [a and b for a,b in zip (one,two)]
    comb = sum(selover)
    v = vvv.venn2(subsets = {'10': sum(one)-comb, '01': sum(two)-comb, '11': comb}, set_labels = labels)
    plt.show()



def simpleheatmap(canvas):
        df = DataFrame(canvas)
        sns.heatmap(df, annot=True)
        
        #df = DataFrame(canvas[:y1map.len,:y2map.len])
        #s= lambda y,x: [ y.getitem[k] for k in x]
        #sns.heatmap(df,annot=True,yticklabels=y1map.itemlist,xticklabels=y2map.itemlist, square=True)
        plt.show()

def doubleheatmap(canvas, cleaned, y1map, y2map, rows, cols, save=None):    

    sns.set(font_scale=1.2,style='white')
    plt.figure(figsize=(12,5))    

    #plt.tight_layout()    
    ax=plt.subplot(121)
    plt.title('Normalized Matches',size=20)
    heatmap(canvas,y1map,y2map,rows,cols, show=False)
    ax=plt.subplot(122)
    plt.title('Processed Matrix',size=20)
    heatmap(cleaned,y1map,y2map,rows,cols, show=False)
    if save:
        plt.tight_layout()
        plt.savefig(save, dpi=300)
    plt.show()

def quickdoubleheatmap(comp1,comp2, save=None):    

    sns.set(font_scale=1.2,style='white')
    plt.figure(figsize=(12,5))    
    #plt.tight_layout()    
    ax=plt.subplot(121)
    plt.title('Clustering1',size=20)
    heatmap(*comp1,show=False)
    ax=plt.subplot(122)
    plt.title('Clustering2',size=20)
    heatmap(*comp2,show=False)
    if save:
        plt.tight_layout()
        plt.savefig(save, dpi=300)
    plt.show()

    
    

from lmz import grouper 
def radviz_sort_features(matrix, reduce=4):

    Agg = agg(n_clusters=None,distance_threshold=0)
    #m.preprocess(500)
    #data = m.a.X.todense()
    Agg.fit(matrix.T)
    sorted_ft = [ a for a in Agg.children_.flatten() if a < matrix.shape[1]]
    if reduce: sorted_ft = [g[0] for g in grouper(sorted_ft,reduce)]
    return matrix[:, sorted_ft]

def radviz(matrix,classes, sort_ft= True, reduce=4):
    if sort_ft: 
        matrix= radviz_sort_features(matrix, reduce=reduce)
    df = pd.DataFrame(matrix)
    #df.boxplot()
    
    df['class']=classes
    size=10 # default is too large
    pd.plotting.radviz(df,'class',s=size, colormap=plt.cm.get_cmap('tab20'))
    plt.show()

def heatmap(canvas,y1map,y2map,row_ind,col_ind, show=True):
        # there is a version that sorts the hits to the diagonal in util/bad... 
        paper = True
        
        sorting = sorted(zip(col_ind, row_ind))
        col_ind, row_ind= list(zip(*sorting))
        
        # some rows are unused by the matching,but we still want to show them:
        order= list(row_ind) + list(set(y1map.integerlist)-set(row_ind) )
        canvas = canvas[order]

        xlabels = y2map.itemlist 
        ylabels = [y1map.getitem[r]for r in order]

        df = DataFrame(-canvas)

        if paper:
            sns.heatmap(df,xticklabels=xlabels,yticklabels=ylabels, annot=False ,linewidths=.5,cmap="YlGnBu" , square=True)
            plt.xlabel('Clusters data set 2')
            plt.ylabel('Clusters data set 1')
        else:
            sns.heatmap(df,xticklabels=ylabels,yticklabels=xlabels, annot=True, square=True)
        #df = DataFrame(canvas[:y1map.len,:y2map.len])
        #s= lambda y,x: [ y.getitem[k] for k in x]
        #sns.heatmap(df,annot=True,yticklabels=y1map.itemlist,xticklabels=y2map.itemlist, square=True)
        if show:
            plt.show()
            
def distrgrid(distances,Y1,Y2,hungmatch):
    # we should make a table first... 
    row_ind, col_ind = hungmatch
    rows=[ (Y1[r],Y2[c],distances[r,c]) for r,c in zip(row_ind,col_ind)]
    #g = sns.FacetGrid(DataFrame(rows,columns=['set1','set2','dist'] ), row="set1",col="set2")
    g = sns.FacetGrid(DataFrame(rows,columns=['set1','set2','dist'] ), row="set1",hue="set2", aspect=5,palette=col)
    g.map(sns.distplot, "dist", hist=False, rug=True);
    g.add_legend();
    plt.ylim(0, 1)
    plt.show()
    g = sns.FacetGrid(DataFrame(rows,columns=['set1','set2','dist'] ), row="set2",hue="set1", aspect=5,palette=col)
    g.map(sns.distplot, "dist", hist=False, rug=True);
    g.add_legend();
    plt.ylim(0, 1)
    plt.show()



def sankey(canvasbackup ,y1map, y2map):

    from pysankey import sankey as pysankey
    # this is the actual drawing
    flow= [(y1map.getitem[a],y2map.getitem[b],-canvasbackup[a][b])
                for a in range(canvasbackup.shape[0])
                    for b in range(canvasbackup.shape[1]) if -canvasbackup[a][b] > 0]

    flow.sort(key= lambda x: x[2],reverse=True)

    df = DataFrame(flow, columns=['set 1','set 2','matches'])
    #from basics import dumpfile
    #dumpfile(df,"adong")
    pysankey(
        left=df['set 1'],
        right=df['set 2'],
        rightWeight=df['matches'],
        #leftLabels = y1map.itemlist,
        #rightLabels = y2map.itemlist,
        leftWeight=df['matches'], 
        aspect=20,
        fontsize=20,
        figureName="Matching",
        colorDict= col
    )


def dreibeidrei(bins,cnt=3): 
    for i,e in enumerate(bins):
            plt.subplot(cnt,cnt,i+1)
            plt.bar(list(range(len(e))),e)
    plt.show()
    

#import pprint
def get_centers(zy,cnt=3):
    mod = NearestNeighbors(n_neighbors=1).fit(zy)
    X,Y = get_centers_1d(zy[:,0], cnt = cnt), get_centers_1d(zy[:,1], cnt=cnt)
    Y.reverse()
    #search = [(x,y) for x in X for y in Y]
    search = [(x,y) for y in Y for x in X]
    #print("search:")
    #pprint.pprint(search)
    distances, indices = mod.kneighbors(search)
    return indices.flatten()
            
            
def get_centers_1d(li,cnt = 3):
    dist = li.max()-li.min()
    dist/=  cnt*2
    #print("asd",li.min(),li.max(), dist)
    return [(li.min()+(a*2+1)*dist) for a in range(cnt)]
    


def tinyumap(X,Y,
        title="No title",
        title_size=10,
        acc : "y:str_description"={}, 
        markerscale=4,
        getmarker = lambda cla: {"marker":'o'},
        col=col,
        label=None,
        alpha = None, 
        legend = False, 
        size=None):
#    print(X)
#    print(Y)
    plt.title(title, size=title_size)
    Y=np.array(Y)
    size=  max( int(4000/Y.shape[0]), 1) if not size else size
    embed = X
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.gca().axes.xaxis.set_ticklabels([])
    for cla in np.unique(Y):
        plt.scatter(embed[Y==cla, 0],
                    embed[Y==cla, 1],
                    color= col[cla],
                    s=size,
                    edgecolors = 'none',
                    alpha = alpha, 
                    label=str(cla), **getmarker(cla)) #str(cla)+" "+acc.get(cla,''),**getmarker(col[cla]))
    #plt.axis('off')
    #plt.xlabel('UMAP 2')
    #plt.ylabel('UMAP 1')
    if legend: 
        plt.legend(markerscale=2,ncol=2,bbox_to_anchor=(1, -.12) )

class tinyUmap(): 

    def __init__(self, dim=(3,3), size= 2):
        figs = (size*dim[1], size*dim[0])

        plt.figure( figsize=figs, dpi=300)
        self.i =0
        self.dim = dim
    

    def next(self): 
        self.i= self.i+1 
        plt.subplot(*self.dim,self.i)

    def draw(self, *a, **b): 
        self.next()
        tinyumap(*a,**b)



def auto_tiny(X,Y, wrap = 'auto', grad= False): 
    
    # how should we wrap:
    if wrap == 'auto':
        d = tinyUmap(dim = (1,len(X)))  # default is a row
    else: 
        print ('not implemented, the idea is to put $wrap many in a row')
        # this means initializiung tinyUmap with another dim

    if not grad:
        for x,y in zip(X,Y): 
            d.draw(x,y, title=None) 
        plt.legend(markerscale=1.5,fontsize='small',ncol=int(len(X)*2.5),bbox_to_anchor=(1, -.12) )

    if grad: 
        for x,y in zip(X,Y): 
            d.next()
            plt.scatter(x[:,0], x[:,1], c=y, s=1)

    plt.show()
    

import natto.process.util as util
def distance_debug(m):
    (a,b),di = util.hungarian(*m.dx, debug = True)
    d = di[a,b]
    cmp2_grad(d,d,*m.d2,m.titles, save = False, cmap='viridis')



from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
def dendro(mat, title, fname='none'):
    links = squareform(mat)
    Z = hierarchy.linkage(links, 'single')
    plt.figure()
    hierarchy.dendrogram(Z) # there is a labels = [] parameter here :) 
    plt.xticks(rotation=45)
    plt.title(title)
    if fname != 'none': 
        plt.savefig(fname, dpi=300)
        plt.close()
    return Z


def plot_noise(lines, title,show_std=False, saveas= False):
    plt.figure( figsize=(5, 5), dpi=300)

    for y,std,label in lines:
        if show_std:
            plt.errorbar(range(0,110,10), y, yerr=std, label=f'{label}')
        else:
            plt.plot(range(0,110,10), y, label=f'{label}')

    plt.grid()
    plt.xlabel('noise percentage', size=14)
    plt.ylabel('similarity', size=14)
    plt.legend(loc= 3, prop={'size': 12})
    #plt.title(title, size= 20)
    plt.tight_layout()
    if saveas:
        plt.savefig(f"{saveas}.png")
        plt.savefig(f"{saveas}.pdf")
    plt.show()


