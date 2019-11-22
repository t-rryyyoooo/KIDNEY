#!/bin/bash

#Input
readonly DATA="~/Desktop/data/kits19/case_00"
readonly CT="/imaging.nii.gz"
readonly LABEL="/segmentation.nii.gz"
readonly SAVE="~/Desktop/data/slice/summed_hist_"
#readonly MODEL="/home/vmlab/Desktop/data/model/model_"
readonly WEIGHT="~/Desktop/data/modelweight/best_sum_"


readonly NUMBERS=(173 002 068 133 155 114 090 105 112 175 183 208 029 065 157 162 141 062 031 156 189 135 020 077 000 009 198 036)

readonly ALPHA=(0.0 0.20 0.40 0.60 0.80)
i=0

for alpha in ${ALPHA[@]}
do
    for number in ${NUMBERS[@]}
    do
        for t in $(seq 0 4)
        do 


            weight="${WEIGHT}${alpha}_${t}.hdf5"
            #model=${MODEL}${alpha}.yml
            save="${SAVE}${alpha}/segmentation/${t}/case_00${number}/label.mha"
            ct=${DATA}${number}${CT}
            label=${DATA}${number}${LABEL}

            
            
            if [ $1 -eq $((i%2)) ]; then
            echo $weight
            #echo $model 
            echo $save
            echo $alpha
            echo "GPU ID : $1"

            python3 segmentationUnet3chAtOnce.py $label $ct $model $weight $save $alpha -g $1

            else
            echo $alpha skipped

            fi
        
        done

    done
    let i++
done