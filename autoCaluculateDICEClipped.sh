#!/bin/bash

#Input
readonly TRUE="$HOME/Desktop/data/box"
readonly RESULT="$HOME/Desktop/data/box"
readonly TEXT="$HOME/Desktop/KIDNEY/result"
readonly PREFIX="6ch_6ch_per"


text="${TEXT}/${PREFIX}.txt"

echo ${TRUE}
echo $RESULT
echo $text

python3 --version
python3 caluculateDICEPerClipped.py ${TRUE} ${RESULT} > $text

echo Done
