#!/bin/bash

#Input
readonly TRUE="~/Desktop/data/kits19/case_00"
readonly RESULT="~/Desktop/data/slice/summed_hist_"
#readonly ALPHA=(0.0 0.20 0.40 0.60 0.80 1.0)
readonly ALPHA=(0.0)

for a in ${ALPHA[@]}
do
    for i in $(seq 0 0)
    do 
    
        result="${RESULT}${a}/segmentation/$i/case_00"
        
        echo ${TRUE}
        echo ${result}
        echo "Alpha: $a"

        python3 --version
        python3 caluculateDICE.py ${TRUE} ${result} $a >> "~/Desktop/KIDNEY/result/summed_$a_$i.txt"

  

done
