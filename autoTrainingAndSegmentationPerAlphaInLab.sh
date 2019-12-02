#!/bin/bash

#Input
readonly TRAINING="$HOME/Desktop/data/textList/training_sum_float_"
readonly VALIDATION="$HOME/Desktop/data/textList/validation_sum_float_"
readonly WEIGHT="$HOME/Desktop/data/modelweight/best_sum_float_"
readonly HISTORY="$HOME/Desktop/data/history/history_sum_float_"

readonly DATA="$HOME/Desktop/data/kits19/data"
readonly CT="/imaging.nii.gz"
readonly LABEL="/segmentation.nii.gz"
readonly SAVE="$HOME/Desktop/data/slice/summed_hist_float_"

readonly NUMBERS=(173 002 068 133 155 114 090 105 112 175 183 208 029 065 157 162 141 062 031 156 189 135 020 077 000 009 198 036)
readonly ALPHA=(0.80 1.0)

for alpha in ${ALPHA[@]}
do
    for t in $(seq 0 4)
    do

        training=${TRAINING}${alpha}.txt
        validation=${VALIDATION}${alpha}.txt
        weight="${WEIGHT}${alpha}_${t}.hdf5"
        #model='${MODEL}${alpha}_${t}.yml'
        histories="${HISTORY}${alpha}_${t}.txt"

        
        echo $training
        echo $weight
        echo $validation
        echo $histories
        echo $alpha

        python3 buildUnet3chAugmentation.py ${training} ${weight} -t ${validation} --history ${histories} -b 15  -e 40

        for number in ${NUMBERS[@]}
        do 
            save="${SAVE}${alpha}/segmentation/${t}/case_00${number}/label.mha"
            ct="${DATA}/case_00${number}${CT}"
            label="${DATA}/case_00${number}${LABEL}"

            echo $label
            echo $ct
            echo $weight
            echo $save
            echo $alpha

            python3 segmentationUnet3chAtOnce.py $label $ct $weight $save $alpha

        done
    done

done