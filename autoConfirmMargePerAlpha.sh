#!/bin/bash

#Input
readonly TEXT="$HOME/Desktop/data/textList"
readonly SUFFIX="sum_float_"

readonly ALPHA=(0.0 0.20 0.40 0.60 0.80 1.0)

for a in ${ALPHA[@]}
do 

    
    suffix=${SUFFIX}${a}

    echo $TEXT
    echo $suffix

    python3 confirmMarge.py ${TEXT} ${suffix}

done