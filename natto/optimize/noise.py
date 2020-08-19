from scipy.sparse import csr_matrix 
from natto.process import noise
from natto.out import quality as Q
import matplotlib
matplotlib.use('Agg')




def get_noise_run(args):
    loader, pool, cluster= args
    adat = loader()
    noiserange=  range(0,110,10)
    if type(adat.X) != csr_matrix:
        adat.X = csr_matrix(adat.X)

    # i could use mp in get_noise_data i think 
    rdydata= noise.get_noise_data(adat, noiserange, title="magic", poolsize=pool, cluster=cluster)
    r = [ Q.rari_score(*m.labels, *m.dx) for m in rdydata]
    return r 



def get_noise_run_moar(args):
    loader, cluster, level = args
    if level == 0:
        # this should be 1 and 1, 
        # its not always,
        # i tracked it down to randomization in the projection
        # a few weeks ago
        return (1,1) 

    # loaddata
    adat = loader()
    if type(adat.X) != csr_matrix:
        adat.X = csr_matrix(adat.X)


    #noise and preproc
    
    # i could use mp in get_noise_data i think 
    m =  noise.get_noise_single(adat, level)


    labels = cluster(*m.dx)
    r = Q.rari_score(*labels, *m.dx)
    return r 










