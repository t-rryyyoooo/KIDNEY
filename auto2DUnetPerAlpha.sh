#!/bin/bash

#Input
readonly TRAINING="$HOME/Desktop/tanimoto/data/textList/training_"
readonly VALIDATION="$HOME/Desktop/tanimoto/data/textList/validation_"
readonly MODEL="$HOME/Desktop/tanimoto/data/model/model_"
readonly WEIGHT="$HOME/Desktop/tanimoto/data/weight/best_"
readonly HISTORY="$HOME/Desktop/tanimoto/data/history/history_"

#readonly ALPHA=(0.55 0.60 0.65 0.70 0.80 0.85 0.90 0.95 1.0)
readonly ALPHA=(1.0 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55)

i=0
for alpha in ${ALPHA[@]}
do

    training=${TRAINING}${alpha}.txt
    validation=${VALIDATION}${alpha}.txt
    weight=${WEIGHT}${alpha}.hdf5
    model=${MODEL}${alpha}.yml
    histories=${HISTORY}${alpha}.txt

    
    
    if [ $1 -eq $((i%2)) ]; then
    echo $training
    echo $validation
    echo $weight
    echo $model 
    echo $histories
    echo $alpha
    python3 buildUnet3chAugmentation.py ${training} ${model} -t ${validation} --bestfile ${weight} --history ${histories} -b 15  -e 40 -g $1
    python3 mail.py $histories
    
    else
    echo $alpha skipped

    fi
    let i++

done