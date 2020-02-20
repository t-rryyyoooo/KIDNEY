#!/bin/bash

#Input
readonly TRAINING="$HOME/Desktop/data/textList/training_"
readonly VALIDATION="$HOME/Desktop/data/textList/validation_"
readonly WEIGHT="$HOME/Desktop/data/modelweight/best_"
readonly INITIALWEIGHT="$HOME/Desktop/data/initialWeight/initial_"
readonly HISTORY="$HOME/Desktop/data/history/hostory_"

echo -n Suffix:
read suffix
echo -n "Is the weight file's suffix the same as above?[yes/no]:"
read choice

training="${TRAINING}${suffix}.txt"
validation="${VALIDATION}${suffix}.txt"

if [ $choice = "yes" ]; then
	histories="${HISTORY}${suffix}.txt"
	initialWeight="${INITIALWEIGHT}${suffix}.hdf5"
        weight="${WEIGHT}${suffix}.hdf5"
else
        echo -n suffix:
        read newSuffix

	histories="${HISTORY}${newSuffix}.txt"
	initialWeight="${INITIALWEIGHT}${newSuffix}.hdf5"
        weight="${WEIGHT}${newSuffix}.hdf5"
fi

echo -n GPU_ID:
read id
echo $training
echo $validation
echo $weight
echo $histories

python3 buildUnet3chAugmentation.py ${training} ${weight} ${initialWeight} -t ${validation} --history ${histories} -b 15 -e 40 -g $id

