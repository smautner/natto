import scanpy as sc
from scipy.sparse import csr_matrix
import pandas as pd
from lmz import *
import anndata as ad
import numpy as np


####
# oldest datasets
#######
load = lambda f: [l for l in open(f,'r').read().split('\n') if len(l)>1]


def do_subsample(adata, subsample, seed = None):
    if not subsample:
        return adata

    if subsample <1:
        sc.pp.subsample(adata, fraction=subsample, n_obs=None, random_state=seed, copy=False)
    else:
        if adata.shape[0] < subsample:
            return adata
        sc.pp.subsample(adata, fraction=None, n_obs=subsample, random_state=seed, copy=False)
    return adata

def loadlabels(labels, ids):
    cellid_to_clusterid = {row.split(',')[0]:hash(row.split(',')[1]) for row in labels[1:]} #
    clusterid_to_nuclusterid = {item:clusterid for clusterid, item in enumerate(sorted(list(set(cellid_to_clusterid.values()))))}
    #print (clusterid_to_nuclusterid)
    return np.array([ clusterid_to_nuclusterid.get(cellid_to_clusterid.get(idd[:-2],-1),-1)  for idd in ids])

def filter(adata, cells='mito'):

    if cells == 'seurat':
        adata = adata[adata.obs['labels'] != -1,:] # -1 are unannotated cells, there are quite a lot of these here :)
    elif cells == 'mito':
        mito_genes = adata.var_names.str.startswith('MT-')
        adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
        adata = adata[adata.obs['percent_mito'] < 0.05, :]

    return adata

def load3k(cells: 'mito all seurat' ='mito', subsample=.15, seed = None, pathprefix = '..')-> 'anndata object':
    adata =  sc.read_10x_mtx(
    pathprefix+'/data/3k/hg19/',
    var_names='gene_symbols', cache=True)
    adata.obs['labels']= loadlabels(load( pathprefix+"/data/3k/pbmc.3k.labels"), load(pathprefix+"/data/3k/hg19/barcodes.tsv"))
    adata = filter(adata,cells)
    adata = do_subsample(adata, subsample,seed)
    return adata

def load6k(cells: 'mito all seurat' ='mito', subsample=.25, seed=None, pathprefix = '..')-> 'anndata object':
    adata =  sc.read_10x_mtx(
    pathprefix+'/data/6k/hg19/',
    var_names='gene_symbols', cache=True)

    adata.obs['labels']= loadlabels(load( pathprefix+"/data/6k/pbmc.6k.labels"), load( pathprefix+"/data/6k/hg19/barcodes.tsv"))

    adata = filter(adata,cells)
    adata = do_subsample(adata, subsample, seed)
    return adata



def loadpbmc(path=None, subsample=None, seed=None):
    adata = sc.read_10x_mtx( path,  var_names='gene_symbols', cache=True)
    adata = do_subsample(adata, subsample,seed)
    return adata

def load3k6k(subsample=False,seed=None, pathprefix = '..'):
    return load3k(subsample=subsample, seed=seed, pathprefix = pathprefix), load6k(subsample=subsample,seed=seed, pathprefix = pathprefix)

def loadp7de(subsample=False,pathprefix='..', seed=None):
    return loadpbmc('%s/data/p7d'%pathprefix ,subsample,seed), loadpbmc('%s/data/p7e'%pathprefix,subsample, seed)

def load4k8k(subsample=False,pathprefix='..',seed=None):
    return loadpbmc('%s/data/4k'% pathprefix,subsample, seed=seed), loadpbmc('%s/data/8k'%pathprefix,subsample,seed=seed)

def loadimmune(subsample=False, pathprefix='..',seed=None):
    return loadpbmc('%s/data/immune_stim/8'% pathprefix,subsample,seed=seed),\
           loadpbmc('%s/data/immune_stim/9'%pathprefix,subsample,seed=seed)


###
# grunreader
####

def loadgruen_single(path,subsample):
    mtx_path = path+".1.counts_raw.csv.gz"
    things = pd.read_csv(mtx_path, sep='\t').T
    adata = ad.AnnData(things)

    truthpath = path+".cell_assignments.csv.gz"
    truth  = pd.read_csv(truthpath, sep='\t')
    #adata.obs['true']  = list(truth['assigned_cluster'])
    adata.obs['true']  = list(truth['celltype'])


    do_subsample(adata, subsample)
    return adata

def loadgruen(subsample=False, pathprefix='..', methods=['human1','human2']):
    return [loadgruen_single('%s/data/punk/%s'% (pathprefix,method),subsample) for method in methods]



########
# LOAD DCA
#########

from anndata import read_h5ad

def loaddca_h5(path,subsample):
    dca_path = path+"/adata.h5ad"
    adata = read_h5ad(dca_path)
    adata.obsm['tsv'] =  loaddca_late(path)

    do_subsample(adata, subsample)

    return adata

def loaddca_late(path,subsample=None):
    dca_path = path+"/latent.tsv"
    things = pd.read_csv(dca_path, index_col=0, header = None, sep='\t')
    return things
    '''
    adata = ad.AnnData(things)
    do_subsample(adata, subsample)
    return adata
    '''

#  h1  h2  h3  h4  immuneA  immuneB  m3k  m4k  m6k  m8k  p7d  p7e
def loaddca(fname, subsample = False, latent = False):
    path = "../data/dca/"+fname
    if latent:
        adata = loaddca_late(path, subsample)
    else:
        adata = loaddca_h5(path, subsample)
    return adata

def loaddca_p7de(subsample):
    return loaddca("p7d", subsample), loaddca("p7e", subsample)
def loaddca_l_p7de(subsample):
    return loaddca("p7d", subsample, latent=True), loaddca("p7e", subsample,latent=True)

def loaddca_immuneab(subsample):
    return loaddca("immuneA", subsample), loaddca("immuneB", subsample)
def loaddca_l_immuneab(subsample):
    return loaddca("immuneA", subsample, latent= True), loaddca("immuneB", subsample, latent=True)

def loaddca_3k6k(subsample):
    return loaddca("m3k", subsample), loaddca("m6k", subsample)

def loaddca_l_3k6k(subsample):
    return loaddca("m3k", subsample, latent= True), loaddca("m6k", subsample, latent=True)


def loaddca_h1h2(subsample):
    return loaddca("h1", subsample), loaddca("h2", subsample)




###
# artificial data
#########

datanames = ["default","even",'si1','si2','c7','c6','c5','facl0.05','facl2']

def loadgalaxy(s=True,subsample=False):
    for i in range(1,10): # 1..9 ,,, 10 is below
        yield loadarti('../data/galaxy',f"fac{'s' if s else 'l'}0_{i}.",subsample=subsample)
    yield loadarti('../data/galaxy',f"fac{'s' if s else 'l'}1_0.",subsample=subsample)




def loadarti(path, dataname,subsample=False):

    batch = open(path+"/%sbatch.art.csv" % dataname, "r").readlines()[1:]
    batch = np.array([ int(x[-2]) for x in batch])

    cnts = open(path+"/%scounts.art.csv" % dataname, "r").readlines()[1:]

    cnts = [ Map(int,line.split(',')[1:])  for line in cnts]
    #cnts= sp.sparse.csr_matrix(Transpose(cnts))
    cnts= np.matrix(Transpose(cnts))

    a = ad.AnnData( cnts[batch == 1] )
    b = ad.AnnData( cnts[batch == 2] )
    b.var['gene_ids'] = {i:i for i in range(cnts.shape[1])}
    a.var['gene_ids'] = {i:i for i in range(cnts.shape[1])}
    a= do_subsample(a, subsample)
    b = do_subsample(b, subsample)

    return a,b


def loadarti_truth(path, p1, p2, dataname):
    grp = open(path+"/%sgroup.art.csv" % dataname, "r").readlines()[1:]
    grp = np.array([ int(x[-2]) for x in grp])

    batch = open(path+"/%sbatch.art.csv" % dataname, "r").readlines()[1:]
    batch = np.array([ int(x[-2]) for x in batch])


    gr1, gr2 =  grp[batch == 1 ], grp[batch==2]

    return gr1[p1], gr2[p2]



#######################3
# LOADING 100 DATASETS
######################
def get100names(path = '../data/100/data'):
    return Map(lambda x: x.strip(), open(path+"/lol2.txt","r").readlines())

def get57names():
    # subset of the 100 datasets...
    # datasets have more than 5000 cells and there are at least 4 examples per cluster, there are 8 clusters
    return ['Bonemarrow_10xchromium_SRA779509-SRS3805245_6248', 'Bonemarrow_10xchromium_SRA779509-SRS3805246_6024', 'Bonemarrow_10xchromium_SRA779509-SRS3805247_7623', 'Bonemarrow_10xchromium_SRA779509-SRS3805248_6395', 'Bonemarrow_10xchromium_SRA779509-SRS3805255_7559', 'Bonemarrow_10xchromium_SRA779509-SRS3805258_5683', 'Bonemarrow_10xchromium_SRA779509-SRS3805262_6431', 'Bonemarrow_10xchromium_SRA779509-SRS3805266_6210', 'Kidneyorganoids_drop-seq_SRA652805-SRS2870733_2290', 'Kidneyorganoids_drop-seq_SRA652805-SRS2870734_2038', 'Kidneyorganoids_drop-seq_SRA652805-SRS2870735_1761', 'Kidneyorganoids_drop-seq_SRA652805-SRS2870743_1470', 'Liver_10xchromium_SRA716608-SRS3391629_4190', 'Liver_10xchromium_SRA716608-SRS3391630_4719', 'Liver_10xchromium_SRA716608-SRS3391632_6158', 'Liver_10xchromium_SRA716608-SRS3391633_6806', 'Placenta_10xchromium_SRA782908-SRS3815597_3618', 'Placenta_10xchromium_SRA782908-SRS3815600_7205', 'Placenta_drop-seq_SRA782908-SRS3815594_391', 'Placenta_drop-seq_SRA782908-SRS3815595_392', 'Placenta_drop-seq_SRA782908-SRS3815596_292', 'Placenta_drop-seq_SRA782908-SRS3815598_301', 'Placenta_drop-seq_SRA782908-SRS3815599_2189', 'Placenta_drop-seq_SRA782908-SRS3815601_614', 'Placenta_drop-seq_SRA782908-SRS3815602_548', 'Placenta_drop-seq_SRA782908-SRS3815603_247', 'Prostate_10xchromium_SRA742961-SRS3565195_6101', 'Prostate_10xchromium_SRA742961-SRS3565196_7789', 'Prostate_10xchromium_SRA742961-SRS3565197_11468', 'Prostate_10xchromium_SRA742961-SRS3565198_7718', 'Prostate_10xchromium_SRA742961-SRS3565199_8069', 'Prostate_10xchromium_SRA742961-SRS3565201_7479', 'Prostate_10xchromium_SRA742961-SRS3565203_6453', 'Prostate_10xchromium_SRA742961-SRS3565206_6950', 'Prostate_10xchromium_SRA742961-SRS3565208_8538', 'Prostate_10xchromium_SRA742961-SRS3565211_6264', 'Tcells_10xchromium_SRA665712-SRS3034950_10979', 'Tcells_10xchromium_SRA665712-SRS3034951_14332', 'Tcells_10xchromium_SRA665712-SRS3034952_7788', 'Tcells_10xchromium_SRA665712-SRS3034953_12974', 'Testicle_10xchromium_SRA667709-SRS3065427_3045', 'Testicle_10xchromium_SRA667709-SRS3065429_3066', 'Testicle_10xchromium_SRA667709-SRS3065431_3586', 'Testis_10xchromium_SRA645804-SRS2823404_4197', 'Testis_10xchromium_SRA645804-SRS2823405_3598', 'Testis_10xchromium_SRA645804-SRS2823406_3989', 'Testis_10xchromium_SRA645804-SRS2823410_4045', 'Testis_10xchromium_SRA645804-SRS2823411_2167', 'Testis_10xchromium_SRA645804-SRS2823412_5299', 'Tumor_10xchromium_SRA658915-SRS2944101_906', 'Tumor_10xchromium_SRA658915-SRS2944102_1229', 'Tumor_10xchromium_SRA658915-SRS2944103_865', 'Tumor_10xchromium_SRA658915-SRS2944104_1031', 'Tumor_10xchromium_SRA658915-SRS2944109_993', 'Tumor_10xchromium_SRA658915-SRS2944110_939', 'Tumor_10xchromium_SRA658915-SRS2944111_1197', 'Tumor_10xchromium_SRA658915-SRS2944112_1242']

def get71names():
    return ['Bonemarrow_10xchromium_SRA779509-SRS3805245_6248', 'Bonemarrow_10xchromium_SRA779509-SRS3805246_6024', 'Bonemarrow_10xchromium_SRA779509-SRS3805247_7623', 'Bonemarrow_10xchromium_SRA779509-SRS3805248_6395', 'Bonemarrow_10xchromium_SRA779509-SRS3805255_7559', 'Bonemarrow_10xchromium_SRA779509-SRS3805258_5683', 'Bonemarrow_10xchromium_SRA779509-SRS3805262_6431', 'Bonemarrow_10xchromium_SRA779509-SRS3805266_6210', 'Colon_10xchromium_SRA703206-SRS3296611_4826', 'Colon_10xchromium_SRA703206-SRS3296612_6476', 'Colon_10xchromium_SRA728025-SRS3454422_2322', 'Colon_10xchromium_SRA728025-SRS3454423_2283', 'Colon_10xchromium_SRA728025-SRS3454424_1898', 'Colon_10xchromium_SRA728025-SRS3454425_3429', 'Colon_10xchromium_SRA728025-SRS3454426_3423', 'Colon_10xchromium_SRA728025-SRS3454427_3840', 'Colon_10xchromium_SRA728025-SRS3454428_5459', 'Colon_10xchromium_SRA728025-SRS3454430_4910', 'Cordblood_10xchromium_SRA769148-SRS3747193_2674', 'Cordblood_10xchromium_SRA769148-SRS3747194_2581', 'Cordblood_10xchromium_SRA769148-SRS3747195_2160', 'Cordblood_10xchromium_SRA769148-SRS3747196_2292', 'Cordblood_10xchromium_SRA769148-SRS3747197_2500', 'Cordblood_10xchromium_SRA769148-SRS3747198_3906', 'Kaposissarcoma_10xchromium_SRA843432-SRS4322339_2914', 'Kaposissarcoma_10xchromium_SRA843432-SRS4322341_3493', 'Kaposissarcoma_10xchromium_SRA843432-SRS4322342_3372', 'Kaposissarcoma_10xchromium_SRA843432-SRS4322343_3940', 'Kaposissarcoma_10xchromium_SRA843432-SRS4322345_3718', 'Kaposissarcoma_10xchromium_SRA843432-SRS4322346_4479', 'Kidneyorganoids_drop-seq_SRA652805-SRS2870733_2290', 'Kidneyorganoids_drop-seq_SRA652805-SRS2870734_2038', 'Kidneyorganoids_drop-seq_SRA652805-SRS2870735_1761', 'Kidneyorganoids_drop-seq_SRA652805-SRS2870743_1470', 'Ovariantumor_10xchromium_SRA634975-SRS2724911_2547', 'Ovariantumor_10xchromium_SRA634975-SRS2724912_2139', 'Ovariantumor_10xchromium_SRA634975-SRS2724913_2262', 'Ovariantumor_10xchromium_SRA634975-SRS2724914_2094', 'Prostate_10xchromium_SRA742961-SRS3565195_6101', 'Prostate_10xchromium_SRA742961-SRS3565196_7789', 'Prostate_10xchromium_SRA742961-SRS3565197_11468', 'Prostate_10xchromium_SRA742961-SRS3565198_7718', 'Prostate_10xchromium_SRA742961-SRS3565199_8069', 'Prostate_10xchromium_SRA742961-SRS3565201_7479', 'Prostate_10xchromium_SRA742961-SRS3565203_6453', 'Prostate_10xchromium_SRA742961-SRS3565206_6950', 'Prostate_10xchromium_SRA742961-SRS3565208_8538', 'Prostate_10xchromium_SRA742961-SRS3565211_6264', 'Tcells_10xchromium_SRA665712-SRS3034950_10979', 'Tcells_10xchromium_SRA665712-SRS3034951_14332', 'Tcells_10xchromium_SRA665712-SRS3034952_7788', 'Tcells_10xchromium_SRA665712-SRS3034953_12974', 'Tcells_10xchromium_SRA794656-SRS3937924_1878', 'Tcells_10xchromium_SRA794656-SRS3937926_1289', 'Tcells_10xchromium_SRA814476-SRS4073850_1975', 'Tcells_drop-seq_SRA867342-SRS4550171_2794', 'Tcells_drop-seq_SRA867342-SRS4550172_4068', 'Tcells_drop-seq_SRA867342-SRS4550173_2768', 'Testicle_10xchromium_SRA667709-SRS3065426_2777', 'Testicle_10xchromium_SRA667709-SRS3065427_3045', 'Testicle_10xchromium_SRA667709-SRS3065428_3007', 'Testicle_10xchromium_SRA667709-SRS3065430_4020', 'Testis_10xchromium_SRA645804-SRS2823404_4197', 'Testis_10xchromium_SRA645804-SRS2823405_3598', 'Testis_10xchromium_SRA645804-SRS2823406_3989', 'Testis_10xchromium_SRA645804-SRS2823407_4046', 'Testis_10xchromium_SRA645804-SRS2823408_4306', 'Testis_10xchromium_SRA645804-SRS2823409_4791', 'Testis_10xchromium_SRA645804-SRS2823410_4045', 'Testis_10xchromium_SRA645804-SRS2823412_5299', 'Testis_10xchromium_SRA645804-SRS3572594_4574']

def get100gz(item, path = '../data/100/data'):
    mtx_path = f"{path}/{item}.counts.gz"
    print("load path: ", mtx_path)
    things = pd.read_csv(mtx_path, sep='\t').T
    adata = ad.AnnData(things)
    adata.X=csr_matrix(adata.X)
    return adata
    #truthpath = path+".cell_assignments.csv.gz"
    #truth  = pd.read_csv(truthpath, sep='\t')
    #adata.obs['true']  = list(truth['assigned_cluster'])
    #adata.obs['true']  = list(truth['celltype'])

def load100(item, path='../data/100/data', seed= None, subsample=None, remove_unlabeled = False):
    adata =  ad.read_h5ad(f"{path}/{item}.h5")

    if remove_unlabeled:
        adata = adata[adata.obs['true']!=-1]

    i = adata.X.shape
    if subsample:
        sc.pp.subsample(adata, fraction=None, n_obs=subsample, random_state=seed, copy=False)
    print(f"LOADING: SHAPE {i} (subsample)-> {adata.X.shape}")
    return adata




def load100addtruthAndWrite(adata,item, path='../data/100/data'):
    fname = f"{path}/{item}.cluster.txt"
    lol = open(fname,'r').readlines()


    barcode_cid={}
    for line in lol:
        bc,cl =  line.strip().split()
        barcode_cid[bc]= int(cl)

    #adata = ad.read_h5ad(fname)
    adata.obs['true'] = [barcode_cid.get(a,-1)  for a in adata.obs.index]
    fname = f"{path}/{item}.h5"
    adata.write(fname, compression='gzip')





