#!/bin/bash

#Input
readonly SLICE="$HOME/Desktop/data/slice/6ch"
readonly SAVE="$HOME/Desktop/data/textList"
readonly SUFFIX=""


sliceOriginal="${SLICE}/original/path"
sliceTransform="${SLICE}/transform/path"
suffixOriginal="${SUFFIX}original"
suffixTransform="${SUFFIX}transform"


echo $sliceOriginal
echo $suffixOriginal

python3 merge.py ${sliceOriginal} ${SAVE} ${suffixOriginal}

echo $sliceTransform
echo $suffixTransform

python3 merge.py ${sliceTransform} ${SAVE} ${suffixTransform}

