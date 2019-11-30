#!/bin/bash

#Input
readonly DATA="$HOME/Desktop/data/kits19/case_00"
readonly CT="/imaging.nii.gz"
readonly LABEL="/segmentation.nii.gz"
readonly SAVE="$HOME/Desktop/data/slice/summed_hist_float_"
readonly WEIGHT="$HOME/Desktop/data/modelweight/best_sum_float_"
readonly NPY="$HOME/Desktop/KIDNEY/sumHistFloat.npy"

readonly NUMBERS=(000 002 009 020 029 031 036 062 065 068 077 090 105 112 114 133 135 141 155 156 157 162 173 175 183 189 198 208)
readonly ALPHA=(0.0 0.20 0.40 0.60)
i=0

for alpha in ${ALPHA[@]}
do
    for number in ${NUMBERS[@]}
    do
        for t in $(seq 0 4)
        do 


            weight="${WEIGHT}${alpha}_${t}.hdf5"
            save="${SAVE}${alpha}/segmentation/${t}/case_00${number}/label.mha"
            ct=${DATA}${number}${CT}
            label=${DATA}${number}${LABEL}

            
            
            if [ $1 -eq $((i%2)) ]; then
            echo $weight
            echo $save
            echo $alpha
            echo "GPU ID : $1"

            python3 segmentationUnet3chAtOnceSummedHistFloat.py $label $ct $weight $save $NPY $alpha -g $1

            else
            echo $alpha skipped

            fi
        
        done

    done
    let i++
done