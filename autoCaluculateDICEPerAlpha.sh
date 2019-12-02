#!/bin/bash

#Input
readonly TRUE="$HOME/Desktop/data/kits19"
readonly RESULT="$HOME/Desktop/data/slice/summed_hist_float_"
#readonly ALPHA=(0.0 0.20 0.40 0.60 0.80 1.0)
readonly TEXT="$HOME/Desktop/KIDNEY/result/"
readonly PREFIX="summed_float_"
#readonly ALPHA=(0.0)
ALPHA=($@)
for a in ${ALPHA[@]}
do
    for i in $(seq 0 4)
    do 
    
        results="${RESULT}${a}/segmentation/$i"
        text="${TEXT}${PREFIX}${a}_${i}.txt"

        echo ${TRUE}
        echo ${results}
        echo "Alpha: $a"
        echo $text

        python3 --version
        python3 caluculateDICE.py ${TRUE} ${results}> $text

        echo Done
    done
done
