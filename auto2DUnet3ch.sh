#!/bin/bash

#Input
readonly TRAINING="$HOME/Desktop/data/textList/training_"
readonly VALIDATION="$HOME/Desktop/data/textList/validation_"
readonly WEIGHT="$HOME/Desktop/data/modelweight/best_"
readonly HISTORY="$HOME/Desktop/data/history/hostory_"

echo -n Suffix:
read suffix
echo -n GPU_ID:
read id

training="${TRAINING}${suffix}.txt"
validation="${VALIDATION}${suffix}.txt"
weight="${WEIGHT}${suffix}.hdf5"
histories="${HISTORY}${suffix}.txt"
echo $training
echo $validation
echo $weight
echo $histories

python3 buildUnet3chAugmentation.py ${training} ${weight} -t ${validation} --history ${histories} -b 15 -e 40 -g $id

