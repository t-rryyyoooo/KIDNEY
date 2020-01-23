#!/bin/bash

#Input
readonly TRUE="$HOME/Desktop/data/box/AABB/nonBlackNoReverse"
readonly RESULT="$HOME/Desktop/data/box/AABB/nonBlackNoReverse"
readonly TEXT="$HOME/Desktop/KIDNEY/result"
readonly PREFIX="nonBlackNoReverse"


text="${TEXT}/${PREFIX}.txt"

echo ${TRUE}
echo $RESULT
echo $text

python3 --version
python3 caluculateDICEClipped.py ${TRUE} ${RESULT} > $text

echo Done
