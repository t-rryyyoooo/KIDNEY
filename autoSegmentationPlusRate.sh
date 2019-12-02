#!/bin/bash

#Input
readonly DATA="/home/kakeya/Desktop/tanimoto/data/kits19"
readonly CT="/imaging.nii.gz"
readonly LABEL="/segmentation.nii.gz"
readonly SAVE="/home/kakeya/Desktop/tanimoto/data/slice/hist_"
readonly MODEL="/home/kakeya/Desktop/tanimoto/data/model/model_"
readonly WEIGHT="/home/kakeya/Desktop/tanimoto/data/weight/best_"


readonly NUMBERS=(000 002 009 020 029 031 036 062 065 068 077 090 105 112 114 133 135 141 155 156 157 162 173 175 183 189 198 208)

readonly ALPHA=(0.00 0.20 0.40 )

for alpha in ${ALPHA[@]}
do
    for number in ${NUMBERS[@]}
    do
        for t in $(seq 0 4)
        do 


            weight=${WEIGHT}${alpha}.hdf5
            #model=${MODEL}${alpha}.yml
            model='/home/kakeya/Downloads/2DUnetModel_re0.30.yml'
            save="${SAVE}${alpha}/segmentation/case_00${number}/label.mha"
            ct="${DATA}/case_00${number}${CT}"
            label="${DATA}/case_00${number}${LABEL}"

            
            
            if [ $1 -eq $((i%2)) ]; then
            echo $weight
            echo $model 
            echo $save
            echo $alpha

            python3 segmentationUnet3chAtOnce.py $label $ct $model $weight $save $alpha -g $1

            else
            echo $alpha skipped

            fi
        
        done

    done
    let i++
done