#!/bin/bash

#Input
readonly DATA="$HOME/Desktop/data/kits19"
readonly CT="/imaging.nii.gz"
readonly LABEL="/segmentation.nii.gz"
readonly SAVE="$HOME/Desktop/data/slice/newOriginal/segmentation0"
readonly WEIGHT="$HOME/Desktop/data/modelweight/best_3ch_selected.hdf5"

NUMBERS=(019 023 054 093 096 123 127 136 141 153 188 191 201)
#NUMBERS=(001 017 020 022 043 082 094 115 120 137 173 174 205)
echo -n GPU_ID:
read id
for number in ${NUMBERS[@]}
do


    save="${SAVE}/case_00${number}/label.mha"
    ct="${DATA}/case_00${number}${CT}"
    label="${DATA}/case_00${number}${LABEL}"
    directory="${DATA}/case_00${number}"


    echo $WEIGHT
    echo $save
    echo "GPU ID: $id"

    python3 segmentationUnet3chAtOnce.py $label $ct $WEIGHT $save -g $id


done
