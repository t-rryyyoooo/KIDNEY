#!/bin/bash

#Input
readonly TRAINING="~/Desktop/data/textList/training_sum"
readonly VALIDATION="~/Desktop/data/textList/validation_sum"
readonly WEIGHT="~/Desktop/data/modelweight/best_sum"
readonly HISTORY="~/Desktop/data/history/history_sum"
#readonly MODEL="/home/kakeya/Desktop/tanimoto/data/model/model_"


#readonly ALPHA=(0.0 0.20 0.40 0.60 0.80 1.0)
readonly ALPHA=(0.0 0.20 0.40 0.60 0.80 1.0)
i=0
for alpha in ${ALPHA[@]}
do
    for t in $(seq 0 4)
    do 

        training=${TRAINING}${alpha}.txt
        validation=${VALIDATION}${alpha}.txt
        weight="${WEIGHT}${alpha}_${t}.hdf5"
        histories="${HISTORY}${alpha}_${t}.txt"        

        
        
        if [ $1 -eq $((i%2)) ]; then
        echo $training
        echo $validation
        echo $weight
        #echo $model 
        echo $histories
        echo $alpha
        echo "GPU ID : $1"

        python3 buildUnet3chAugmentation.py ${training} ${weight} -t ${validation} --history ${histories} -b 15  -e 40 -g $1
        #python3 mail.py $histories
        
        else
        echo $alpha skipped

        fi
    
    done
    let i++

done