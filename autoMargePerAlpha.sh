#!/bin/bash

#Input
readonly SLICE="$HOME/Desktop/data/slice/summed_hist_float_"
readonly SAVE="$HOME/Desktop/data/textList"
readonly SUFFIX="sum_float_"

readonly ALPHA=(0.0 0.20 0.40 0.60 0.80 1.0)

for a in ${ALPHA[@]}
do 

    slice="${SLICE}$a/path/case_00"
    suffix=${SUFFIX}${a}

    echo $slice
    echo $suffix

    python3 marge.py ${slice} ${SAVE} ${suffix}

done