#!/bin/bash

#Input
readonly TRAINING="$HOME/Desktop/data/textList/training_"
readonly VALIDATION="$HOME/Desktop/data/textList/validation_"
readonly WEIGHT="$HOME/Desktop/data/modelweight/best_"
readonly HISTORY="$HOME/Desktop/data/history/history_"

echo -n original_suffix:
read originalSuffix
echo -n ref_suffix:
read refSuffix
echo -n weight_suffix:
read weightSuffix
echo -n GPU_ID:
read id

training1="${TRAINING}${originalSuffix}.txt"
training2="${TRAINING}${refSuffix}.txt"
validation1="${VALIDATION}${originalSuffix}.txt"
validation2="${VALIDATION}${refSuffix}.txt"
weight="${WEIGHT}${weightSuffix}.hdf5"
histories="${HISTORY}${weightSuffix}.txt"
echo $training1
echo $training2
echo $validation1
echo $validation2
echo $weight
echo $histories

python3 buildUnet3ch3chAugmentation.py ${training1} ${training2} ${weight} -t ${validation1} ${validation2} --history ${histories} -b 15 -e 40 -g $id

