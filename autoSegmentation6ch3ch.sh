#!/bin/bash

#Input
readonly DATA="$HOME/Desktop/data/box/AABB/nonBlackNoReverse"
readonly SAVE="$HOME/Desktop/data/box/AABB/nonBlackNoReverse"
readonly WEIGHT="$HOME/Desktop/data/modelweight/best_nonBlackNoReverse.hdf5"

NUMBERS=(019 023 054 093 096 123 127 136 141 153 188 191 201)
#NUMBERS=(173 002 068 133 155 114 090 105 112 175 183 208 029 065 157 162 141 062 031 156 189 135 020 077 000 009 198 036)
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
#    python3 segmentationUnet6ch3chAtOnce_copy.py $ctRight $WEIGHT $saveRight -g $id

    echo $ctLeft
    echo $WEIGHT
    echo $saveLeft
    echo "GPU ID: $id"

    python3 segmentationUnet6ch3chAtOnce.py $ctLeft $WEIGHT $saveLeft -g $id
#    python3 segmentationUnet6ch3chAtOnce_copy.py $ctLeft $WEIGHT $saveLeft -g $id


done
