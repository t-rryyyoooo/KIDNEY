#!/bin/bash

#Input
readonly TRAINING="/home/vmlab/Desktop/data/textList/training_"
readonly VALIDATION="/home/vmlab/Desktop/data/textList/validation_"
#readonly MODEL="/home/kakeya/Desktop/tanimoto/data/model/model_"
readonly WEIGHT="/home/vmlab/Desktop/data/modelweight/best_"
readonly HISTORY="/home/vmlab/Desktop/data/history/history_"

#readonly ALPHA=(0.55 0.60 0.65 0.70 0.80 0.85 0.90 0.95 1.0)
readonly ALPHA=(0.00 0.20)

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