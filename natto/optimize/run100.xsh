$MKL_NUM_THREADS =1
$NUMBA_NUM_THREADS =1
$OMP_NUM_THREADS =1
$OPENBLAS_NUM_THREADS =1

import sys
args = sys.argv[1:]
what = args[0]

if what == "run":
    parallel -j 32 --bar --jl job.log ./sim_mtx.py ::: @$(seq 0 99) ::: @$(seq 0 99) ::: @$(seq 0 2)


elif what == "plot":
    #loadblock -d 65 65 5 --diag -m  -f res > ~/distance_fake_multi_nuplacenta.ev
    import loadblock3
    import numpy as np
    res = loadblock3.make_matrix(dim = [6,6,4], fdim =2) 

    spectral = np.nanmean(res[:,:,:,1],axis=2)
    gmm = np.nanmean(res[:,:,:,0],axis=2)

    print(f" i am still alive")
    import matplotlib
    matplotlib.use('module://matplotlib-sixel')
    import plot.dendro as dendro
    dendro.plot(spectral)
    dendro.plot(gmm)


else:
    print("arg should be run or plot")


"""
for i in (seq 0 874)
     for rep in (seq 0 0)
       if ! test -e ./resOMG/(string join _ $i $rep) && echo "$i $rep"; end
end; end |  parallel -j 32 --bar ./combirand_1d.py
#loadblock2 -d 875 11 -f -b 8 resOMG > ~/p3.3.ev
"""






