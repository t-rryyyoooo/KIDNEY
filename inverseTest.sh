#!/bin/bash

# Input
readonly DATA="~/Desktop/data/kits19/case_00"
readonly CT="/imaging.nii.gz"
readonly LABEL="/segmentation.nii.gz"
readonly SAVE="~/Desktop/data/inverseTest/"


readonly NUMBERS=(173 002 068 133 155 114 090 105 112 175 183 208 029 065 157 162 141 062 031 156 189 135 020 077 000 009 198 036)

for i in ${NUMBERS[@]}
do 
    
    labPath=${DATA}${i}${LABEL}
    ctPath=${DATA}${i}${CT}
    savePath="${SAVE}case_00$i"

    python3 inverseTest.py $labPath $ctPath $savePath 1