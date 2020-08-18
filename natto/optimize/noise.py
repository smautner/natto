from scipy.sparse import csr_matrix 
from natto.process import noise
from natto.out import quality as Q
import matplotlib
matplotlib.use('Agg')

def get_noise_run(args):
    loader, pool, cluster= args
    adat = loader()[0]
    noiserange=  range(0,110,20)
    adat.X = csr_matrix(adat.X)

    # i could use mp in get_noise_data i think 
    rdydata= noise.get_noise_data(adat, noiserange, title="should not matter here", poolsize=pool, cluster=cluster)
    r = [ Q.rari_score(*m.labels, *m.dx) for m in rdydata]
    return r 











