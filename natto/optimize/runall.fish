source setvar.fish

for i in (seq 0 $argv[1])
  for j in (seq $i $argv[2])
    for rep in (seq 0 $argv[3])
      if ! test -e ./res/(string join _ $i $j $rep) && echo "$i $j $rep"; end
end; end; end |  parallel -j 32 --bar $argv[4]

