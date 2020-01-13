#!/bin/bash

#Input
readonly DATA="$HOME/Desktop/data/box/OBB"
readonly SAVE="$HOME/Desktop/data/box/OBB/test_6ch"
readonly WEIGHT="$HOME/Desktop/data/modelweight/best_6ch_6ch.hdf5"

NUMBERS=(019 023 054 093 096 123 127 136 141 153 188 191 201)

echo -n GPU_ID:
read id

for number in ${NUMBERS[@]}
do


    save="${SAVE}/case_00${number}"
    ct="${DATA}/case_00${number}"

    saveRight="${save}/segmentation_right.mha"
    sourceRight="${ct}/image_right.nii.gz"
    refRight="${ct}/image_left_transformed.nii.gz"

    saveLeft="${save}/segmentation_left.mha"
    sourceLeft="${ct}/image_left.nii.gz"
    refLeft="${ct}/image_right_transformed.nii.gz"

    echo $sourceRight
    echo $refRight
    echo $WEIGHT
    echo $saveRight
    echo "GPU ID: $id"

    python3 segmentationUnet6ch6chAtOnce_copy.py $sourceRight $refRight $WEIGHT $saveRight -g $id

    echo $sourceLeft
    echo $refLeft
    echo $WEIGHT
    echo $saveLeft
    echo "GPU ID: $id"

    python3 segmentationUnet6ch6chAtOnce_copy.py $sourceLeft $refLeft  $WEIGHT $saveLeft -g $id


done

