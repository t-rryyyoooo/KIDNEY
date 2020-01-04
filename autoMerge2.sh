#!/bin/bash

#Input
readonly SLICE="$HOME/Desktop/data/slice/original"
readonly SAVE="$HOME/Desktop/data/textList"
readonly SUFFIX="host_original"


slice="${SLICE}/path"
suffix=${SUFFIX}

echo $slice
echo $suffix
echo $SAVE

python3 merge.py ${slice} ${SAVE} ${suffix}

