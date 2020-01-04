#!/bin/bash

#Input
readonly DATA="$HOME/Desktop/data/box"
readonly SAVE="$HOME/Desktop/data/box"
readonly WEIGHT="$HOME/Desktop/data/modelweight/best_6ch_3ch.hdf5"

NUMBERS=(019 023 054 093 096 123 127 136 141 153 188 191 201)

echo -n GPU_ID:
read id

for number in ${NUMBERS[@]}
do


    save="${SAVE}/case_00${number}"
    ct="${DATA}/case_00${number}"

    saveRight="${save}/segmentation_right.mha"
    ctRight="${ct}/image_right.nii.gz"

    saveLeft="${save}/segmentation_left.mha"
    ctLeft="${ct}/image_left.nii.gz"

    echo $ctRight
    echo $WEIGHT
    echo $saveRight
    echo "GPU ID: $id"

    python3 segmentationUnet6ch3chAtOnce.py $ctRight $WEIGHT $saveRight -g $id

    echo $ctLeft
    echo $WEIGHT
    echo $saveLeft
    echo "GPU ID: $id"

    python3 segmentationUnet6ch3chAtOnce.py $ctLeft $WEIGHT $saveLeft -g $id


done
