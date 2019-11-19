#!/bin/bash

#Input
readonly DATA="/home/kakeya/Desktop/tanimoto/data/kits19/case_00"
readonly CT="/imaging.nii.gz"
readonly LABEL="/segmentation.nii.gz"
readonly SAVE="/home/kakeya/Desktop/tanimoto/data/slice/hist_"
readonly MODEL="/home/kakeya/Desktop/tanimoto/data/model/model_"
readonly WEIGHT="/home/kakeya/Desktop/tanimoto/data/weight/best_"


readonly NUMBERS=(173 002 068 133 155 114 090 105 112 175 183 208 029 065 157 162 141 062 031 156 189 135 020 077 000 009 198 036)

#readonly ALPHA=(0.55 0.60 0.65 0.70 0.80 0.85 0.90 0.95 1.0)
#readonly ALPHA=(1.0 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55)
readonly ALPHA=(0.00 0.20 0.40 )
i=0
A=0.95
N=173
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
            ct=${DATA}${number}${CT}
            label=${DATA}${number}${LABEL}

            
            
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