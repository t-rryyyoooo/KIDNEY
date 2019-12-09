#!/bin/bash

#Input
readonly TRUE="$HOME/Desktop/data/kits19"
readonly RESULT="$HOME/Desktop/data/slice/hist_"
#readonly ALPHA=(0.0 0.20 0.40 0.60 0.80 1.0)
readonly TEXT="$HOME/Desktop/KIDNEY/result/"
readonly PREFIX="hist_"

echo -n ALPHA=
read ALPHA
for a in ${ALPHA[@]}
do
    for i in $(seq 5 9)
    do 
    
        results="${RESULT}${a}/segmentation/${i}times"
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
