from scipy.sparse import csr_matrix 
from natto.process import noise
from natto.out import quality as Q


from natto.input import load 
loader = lambda: load.loadarti("../data/art", 'si3', subsample= 1000)[0]

def get_noise_run(loader):
    adata = loader()
    noiserange=  range(0,110,10)
    adat.X = csr_matrix(adat.X)

    # i could use mp in get_noise_data i think 
    rdydata= noise.get_noise_data(adat, noiserange, title)
    r = [ Q.rari_score(*m.labels, *m.dx) for m in rdydata]
    return r 











