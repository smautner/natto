source setvar.fish
for i in (seq 0 874)
     for rep in (seq 0 0)
       if ! test -e ./resOMG/(string join _ $i $rep) && echo "$i $rep"; end
end; end |  parallel -j 32 --bar ./combirand_1d.py


#loadblock2 -d 875 11 -f -b 8 resOMG > ~/p3.3.ev




#for rep in (seq 0 2)
#    for i in (seq 0 99)
#        for j in (seq $i 99)
#      ! test -e ./res/(string join _ $i $j $rep) && echo "$i $j $rep";
#    end; end; end |  parallel -j 32 --bar ./sim_mtx.py

#loadblock -d 65 65 5 --diag -m  -f res > ~/distance_fake_multi_nuplacenta.ev
#loadblock3 --mir --fdim 39 > ~/geneselect39.ev







