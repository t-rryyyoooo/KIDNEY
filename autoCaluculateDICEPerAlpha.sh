#!/bin/bash

#Input
readonly TRUE="~/Desktop/data/kits19/case_00"
readonly RESULT="~/Desktop/data/inverseTest/case_00"
readonly ALPHA=(0.0 0.20)

for a in ${ALPHA[@]}
do
    for i in $(seq 0 4)
    do 
    
        result="${RESULT}${a}/segmentation/$i/case_00"
        
        echo ${TRUE}
        echo ${result}
        echo "Alpha: $a"

        python3 --version
        python3 caluculateDICE.py ${TRUE} ${result} $a

  

done
