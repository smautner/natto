


'''
PLAN:
    - we ran CELLVGAE on a dataset.. now we

    1. load the embedding and subsample n cells
    2. load from another dataset n calls and run it through the network
    3. we calculate the simmilarity
    4. dump the result as sim_mtx would
'''


# python sim_NN.py @(datapath+dataset) @(datapath+data) $numcells


import sys
_, nnpath, pathtodata, numcells = sys.argv
numcells = int(numcells)
#print(nnpath,pathtodata, numcells)



# 1.
import ubergauss.tools as t
emb = t.nloadfile(nnpath+'/cellvgae_node_embeddings.npy')
from sklearn.utils import resample
emb = resample(emb,replace = False,n_samples=numcells)
#print(f"{ emb.shape=}")

# 2.1 DUMP SUBSAMPLED TARGET
import anndata
import scanpy as sc
import uuid
target=anndata.read_h5ad(pathtodata+'.h5')
sc.pp.subsample(target, n_obs = numcells,copy=False)
TMPFILE = 'NNTMP/'+str(uuid.uuid4())
target.write(TMPFILE+".h5ad")

# 2.2 run run run
import basics as ba
ba.shexec(f'mkdir {TMPFILE}')
z=f'python -m cellvgae --input_gene_expression_path "{TMPFILE}.h5ad" --hvg 1000 --khvg 250 --graph_type "KNN Scanpy" --k 10 --graph_metric "euclidean" --save_graph --graph_convolution "GAT" --num_hidden_layers 2 --hidden_dims 128 128 --num_heads 3 3 3 3 --dropout 0.4 0.4 0.4 0.4 --latent_dim 10 --load_model_path "{nnpath}/cellvgae_model.pt" --model_save_path "{TMPFILE}" '
ba.shexec(z)

targetemb = t.nloadfile(f"{TMPFILE}/cellvgae_test_node_embeddings.npy")


from natto.process.cluster import gmm_2, spec_2, kmeans_2
from natto.out.quality import rari_score

datapair  = [emb, targetemb]
r=rari_score(*gmm_2(*datapair,nc=15, cov='full'), *datapair)
r2=rari_score(*gmm_2(*datapair,nc=15, cov='tied'), *datapair)

print("distance:",r,r2)
#t.dumpfile([r,r2],"NNTMP/{writeto}")
