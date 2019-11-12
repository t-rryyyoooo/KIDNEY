#!/bin/bash

#Input
readonly TRAINING="/home/kakeya/Desktop/tanimoto/data/textList/training_"
readonly VALIDATION="/home/kakeya/Desktop/tanimoto/data/textList/validation_"
#readonly MODEL="/home/kakeya/Desktop/tanimoto/data/model/model_"
readonly WEIGHT="/home/kakeya/Desktop/tanimoto/data/weight/best_"
readonly HISTORY="/home/kakeya/Desktop/tanimoto/data/history/history_"

#readonly ALPHA=(0.55 0.60 0.65 0.70 0.80 0.85 0.90 0.95 1.0)
readonly ALPHA=(0.40 0.60 0.80 1.00)

i=0
for alpha in ${ALPHA[@]}
do
    for t in $(seq 0 4)
    do 

        training=${TRAINING}${alpha}.txt
        validation=${VALIDATION}${alpha}.txt
        weight="${WEIGHT}${alpha}_${t}.hdf5"
        #model=${MODEL}${alpha}.yml
        histories="${HISTORY}${alpha}_${t}.txt"

        
        
        if [ $1 -eq $((i%2)) ]; then
        echo $training
        echo $validation
        echo $weight
        #echo $model 
        echo $histories
        echo $alpha

        python3 buildUnet3chAugmentation.py ${training} ${model} -t ${validation} --bestfile ${weight} --history ${histories} -b 15  -e 40 -g $1
        #python3 mail.py $histories
        
        else
        echo $alpha skipped

        fi
    
    done
    let i++

done