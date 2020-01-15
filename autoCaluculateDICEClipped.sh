#!/bin/bash

#Input
readonly TRUE="$HOME/Desktop/data/box/AABB/nonBlack"
readonly RESULT="$HOME/Desktop/data/box/AABB/nonBlack"
readonly TEXT="$HOME/Desktop/KIDNEY/result"
readonly PREFIX="notAlignedFull"


text="${TEXT}/${PREFIX}.txt"

echo ${TRUE}
echo $RESULT
echo $text

python3 --version
python3 caluculateDICEClipped.py ${TRUE} ${RESULT} > $text

echo Done
