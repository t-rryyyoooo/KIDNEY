#!/bin/bash

#Input
readonly TRAINING="$HOME/Desktop/data/textList/training_"
readonly VALIDATION="$HOME/Desktop/data/textList/testing_"
readonly WEIGHT="$HOME/Desktop/data/modelweight/best_"
readonly INITIALWEIGHT="$HOME/Desktop/data/initialWeight/initial_"
readonly HISTORY="$HOME/Desktop/data/history/hostory_"

echo -n Suffix:
read suffix
echo -n GPU_ID:
read id

training="${TRAINING}${suffix}.txt"
validation="${VALIDATION}${suffix}.txt"
weight="${WEIGHT}${suffix}.hdf5"
histories="${HISTORY}${suffix}.txt"
initialWeight="${INITIALWEIGHT}${suffix}.hdf5"
echo $training
echo $validation
echo $weight
echo $histories

python3 buildUnet3chAugmentation.py ${training} ${weight} ${initialWeight} -t ${validation} --history ${histories} -b 15 -e 40 -g $id

