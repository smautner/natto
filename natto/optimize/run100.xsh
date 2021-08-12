$MKL_NUM_THREADS =1
$NUMBA_NUM_THREADS =1
$OMP_NUM_THREADS =1
$OPENBLAS_NUM_THREADS =1

import sys
args = sys.argv[1:]
what = args[0]

if what == "run":
    parallel -j 32 --bar ./sim_mtx.py ::: @$(seq 0 5) ::: @$(seq 0 5) ::: @$(seq 0 3)


elif what == "plot":
    #loadblock -d 65 65 5 --diag -m  -f res > ~/distance_fake_multi_nuplacenta.ev
    loadblock3 --mir --fdim 39 > ~/geneselect39.ev

else:
    print("arg should be run or plot")


"""
for i in (seq 0 874)
     for rep in (seq 0 0)
       if ! test -e ./resOMG/(string join _ $i $rep) && echo "$i $rep"; end
end; end |  parallel -j 32 --bar ./combirand_1d.py


#loadblock2 -d 875 11 -f -b 8 resOMG > ~/p3.3.ev





"""






