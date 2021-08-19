$MKL_NUM_THREADS =1
$NUMBA_NUM_THREADS =1
$OMP_NUM_THREADS =1
$OPENBLAS_NUM_THREADS =1

import sys
args = sys.argv[1:]
what = args[0]

import matplotlib
matplotlib.use('module://matplotlib-sixel')
import matplotlib.pyplot as plt

if what == "run":
    parallel -j 32 --bar --jl job.log ./sim_mtx.py ::: @$(seq 0 99) ::: @$(seq 0 99) ::: @$(seq 0 2)


elif what == "plot":
    #loadblock -d 65 65 5 --diag -m  -f res > ~/distance_fake_multi_nuplacenta.ev
    import loadblock3
    import numpy as np
    res = loadblock3.make_matrix(dim = [100,100,3], fdim =2) 

    spectral = np.nanmean(res[:,:,:,1],axis=2)
    gmm = np.nanmean(res[:,:,:,0],axis=2)

    import plot.dendro as dendro
    dendro.plot(spectral)
    dendro.plot(gmm)

elif what  == '100data':
    from natto import input 
    import basics as ba
    dnames = input.get100names(path='../data')
    #sizes = [ input.load100(dna,path = '../data', subsample=False).X.shape[0] for dna in dnames ]
    sizes = [15776, 17768, 12457, 8307, 7932, 10051, 8041, 8724, 7389, 9285, 10549, 8766, 12334, 2322, 2283, 1898, 3429, 3423, 3840, 5459, 4910, 2674, 2581, 2160, 2292, 2500, 3906, 2914, 3493, 3372, 3940, 3718, 4479, 9685, 7826, 6939, 7145, 62811, 76370, 1425, 64887, 6806, 2881, 2300, 2405, 2186, 25985, 9594, 36952, 49696, 32095, 46272, 48421, 40299, 47923, 43451, 6101, 7789, 11468, 7718, 8069, 7479, 6453, 6950, 8538, 6264, 10979, 14332, 7788, 12974, 2376, 1568, 2683, 2794, 4068, 2768, 2777, 24479, 3007, 31425, 4020, 46046, 5389, 5468, 9339, 4046, 4306, 4791, 15101, 56691, 5299, 4574, 19111, 27713, 11493, 5382, 35086, 60199, 58584, 46546]
    size_cut = [s>5000 for s in sizes]
    sizes = ba.np_bool_select(sizes,size_cut)
    dnames = ba.np_bool_select(dnames,size_cut)
    dnames = [d for d in dnames if not d.startswith('Adipo') and not d.startswith("Colon")]
    print(dnames)
    dnames = [d[:5] for d in dnames]
    from collections import Counter
    print(len(dnames))
    print(Counter(dnames))
    plt.hist(sizes, bins = 25)
    plt.show()



else:
    print("arg should be run or plot")


"""
for i in (seq 0 874)
     for rep in (seq 0 0)
       if ! test -e ./resOMG/(string join _ $i $rep) && echo "$i $rep"; end
end; end |  parallel -j 32 --bar ./combirand_1d.py
#loadblock2 -d 875 11 -f -b 8 resOMG > ~/p3.3.ev
"""






