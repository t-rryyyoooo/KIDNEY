#!/bin/bash

#Input
readonly TRUE="$HOME/Desktop/data/kits19"
readonly RESULT="$HOME/Desktop/data/slice/notAligned/segmentation2"
readonly TEXT="$HOME/Desktop/kidney/original/result"
readonly PREFIX="notAligned2"


text="${TEXT}/${PREFIX}.txt"

echo ${TRUE}
echo $RESULT
echo $text

python3 --version
python3 caluculateDICE.py ${TRUE} ${RESULT} > $text

echo Done
