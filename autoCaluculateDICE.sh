#!/bin/bash

#Input
readonly TRUE="$HOME/Desktop/data/kits19"
readonly RESULT="$HOME/Desktop/data/slice/original/segmentation"
readonly TEXT="$HOME/Desktop/KIDNEY/result"
readonly PREFIX="3ch_selected"


text="${TEXT}/${PREFIX}.txt"

echo ${TRUE}
echo $RESULT
echo $text

python3 --version
python3 caluculateDICE.py ${TRUE} ${RESULT} > $text

echo Done
