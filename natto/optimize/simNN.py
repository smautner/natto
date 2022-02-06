
'''
PLAN:
    - we ran CELLVGAE on a dataset.. now we
    1. load the embedding and subsample n cells
    2. load from another dataset n calls and run it through the network
    3. we calculate the simmilarity
    4. dump the result as sim_mtx would
'''




import basics as ba
import sys
import ubergauss.tools as t
from sklearn.utils import resample
import anndata
import scanpy as sc
import uuid
from natto.process.cluster import gmm_2, spec_2, kmeans_2
from natto.out.quality import rari_score
from natto.process import util

print(sys.argv)

_, nnpath, pathtodata, numcells, repeats, id1, id2, python = sys.argv
numcells = int(numcells)
repeats = int(repeats)
#print(nnpath,pathtodata, numcells)

import natto.input as input
alldatasets  = input.get71names()
nnpath += alldatasets[int(id1)]
pathtodata += alldatasets[int(id2)]



def doit(repeat):
    # 1. load our node embedding
    emb = t.nloadfile(nnpath+'/cellvgae_node_embeddings.npy')
    emb = resample(emb,replace = False,n_samples=numcells, random_state = repeat)

    # 2.1 DUMP SUBSAMPLED OTHER DATAA
    target=anndata.read_h5ad(pathtodata+'.h5')
    sc.pp.subsample(target, n_obs = numcells,copy=False,random_state = repeat+31337)
    TMPFILE = 'NNTMP/'+str(uuid.uuid4())
    target.write(TMPFILE+".h5ad")

    # 2.2 run run run;;
    ba.shexec(f'mkdir {TMPFILE}')
    z=f'{python} -m cellvgae --input_gene_expression_path {TMPFILE}.h5ad --hvg 1000 --khvg 250 --graph_type KNNSCANPY --k 10 --graph_metric euclidean --save_graph --graph_convolution GAT --num_hidden_layers 2 --hidden_dims 128 128 --num_heads 3 3 3 3 --dropout 0.4 0.4 0.4 0.4 --latent_dim 10 --load_model_path {nnpath}/cellvgae_model.pt --model_save_path {TMPFILE}/ --genesaveload load --genesaveloadpath  {nnpath}/selected_genes'
    if False:
        z=f'python -m cellvgae --input_gene_expression_path "{TMPFILE}.h5ad" --hvg 1000 --khvg 250 --graph_type "KNN Scanpy" --k 10 --graph_metric "euclidean" --save_graph --graph_convolution "GAT" --num_hidden_layers 2 --hidden_dims 128 128 --num_heads 3 3 3 3 --dropout 0.4 0.4 0.4 0.4 --latent_dim 10 --load_model_path "{nnpath}/cellvgae_model.pt" --model_save_path "{TMPFILE}"'
    asd= ba.shexec(z)
    print("#"*80)
    print(asd)
    print(z)
    print("#"*80)
    ba.jdumpfile(z,f'{TMPFILE}/cmd')
    targetemb = t.nloadfile(f"{TMPFILE}/cellvgae_test_node_embeddings.npy")

    hung,_ = util.hungarian(emb,targetemb)
    targetemb = targetemb[hung[1]]
    datapair  = [emb, targetemb]
    r =rari_score(*gmm_2(*datapair,nc=15, cov='full'), *datapair)
    r2=rari_score(*gmm_2(*datapair,nc=15, cov='tied'), *datapair)
    r3=rari_score(*spec_2(*datapair,nc=15), *datapair)

    print("distance:",r,r2)
    t.dumpfile([r,r2,r3],f"NNOUT/{id1}_{id2}_{repeat}")


for i in range(repeats):
    doit(i)
