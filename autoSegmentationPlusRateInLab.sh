#!/bin/bash

#Input
readonly DATA="~/Desktop/data/kits19/data/case_00"
readonly CT="/imaging.nii.gz"
readonly LABEL="/segmentation.nii.gz"
readonly SAVE="~/Desktop/data/slice/hist_"
#readonly MODEL="/home/vmlab/Desktop/data/model/model_"
readonly WEIGHT="~/Desktop/data/modelweight/best_"


readonly NUMBERS=(173 002 068 133 155 114 090 105 112 175 183 208 029 065 157 162 141 062 031 156 189 135 020 077 000 009 198 036)

readonly ALPHA=(0.00 0.20)
A=0.95
N=173

for alpha in ${ALPHA[@]}
do
    for number in ${NUMBERS[@]}
    do
        for t in $(seq 0 4)
        do 


            weight="${WEIGHT}${alpha}_${t}.hdf5"
            #model='${MODEL}${alpha}_${t}.yml'
            save="${SAVE}${alpha}/segmentation/${t}/case_00${number}/label.mha"
            ct=${DATA}${number}${CT}
            label=${DATA}${number}${LABEL}
      
            echo $weight
            #echo $model 
            echo $save
            echo $alpha

            python3 segmentationUnet3chAtOnce.py $label $ct $weight $save $alpha

        done

    done
done