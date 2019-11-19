#!/bin/bash

#Input
readonly TRUE="~/Desktop/data/kits19/case_00"
readonly RESULT="~/Desktop/data/slice/summed_hist_0.0/label/case_00"
readonly ALPHA=(0.0 0.20)

for a in ${ALPHA[@]}
do
    
    save=${SAVE}${a}
    
    echo ${SLICE}
    echo ${NPY}
    echo ${save}

    python3 --version
    python3 equalizingHistSummed.py ${SLICE} ${NPY} ${save} $a

  

done
