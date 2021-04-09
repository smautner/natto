source setvar.fish
#for i in (seq 0 874)
#    for rep in (seq 5 10)
#      if ! test -e ./res/(string join _ $i $rep) && echo "$i $rep"; end
#end; end |  parallel -j 32 --bar ./combirand_1d.py
#loadblock2 -d 875 11 -f res > ~/point3.1.ev
#mv res res_clusterstuff


for rep in (seq 0 4)
for i in (seq 0 64)
  for j in (seq $i 64)
      ! test -e ./res/(string join _ $i $j $rep) && echo "$i $j $rep";
end; end; end |  parallel -j 32 --bar ./sim_repeat_mtx.py

loadblock -d 65 65 5 --diag -m  -f res > ~/distance_fake_multi_nuplacenta.ev













#for i in (seq 0 $argv[1])
  #for j in (seq $i $argv[2])
    #for rep in (seq 0 $argv[3])
      #if ! test -e ./res/(string join _ $i $j $rep) && echo "$i $j $rep"; end
#end; end; end |  parallel -j 32 --bar $argv[4]
