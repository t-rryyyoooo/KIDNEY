#!/bin/bash

#Input
readonly TRAINING="$HOME/Desktop/data/textList/training_sum_float_"
readonly VALIDATION="$HOME/Desktop/data/textList/validation_sum_float_"
readonly WEIGHT="$HOME/Desktop/data/modelweight/best_sum_float_"
readonly HISTORY="$HOME/Desktop/data/history/history_sum_float_"

#readonly ALPHA=(0.55 0.60 0.65 0.70 0.80 0.85 0.90 0.95 1.0)
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
        echo $validation
        echo $weight
        #echo $model 
        echo $histories
        echo $alpha

        python3 buildUnet3chAugmentation.py ${training} ${weight} -t ${validation} --history ${histories} -b 15  -e 40
        #python3 mail.py $histories
        

    done

done