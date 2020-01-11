#!/bin/bash

#Input
readonly TRAINING="$HOME/Desktop/data/textList/training_host_original.txt"
readonly VALIDATION="$HOME/Desktop/data/textList/validation_host_original.txt"
readonly WEIGHT="$HOME/Desktop/data/modelweight/best_3ch_selected.hdf5"
readonly HISTORY="$HOME/Desktop/data/history/history_3ch_selected.txt"

echo -n GPU_ID:
read id

echo $TRAINING
echo $VALIDATION
echo $WEIGHT
echo $HISTORY

python3 buildUnet3chAugmentation.py ${TRAINING} ${WEIGHT} -t ${VALIDATION} --history ${HISTORY} -b 15 -e 40 -g $id

python3 mail.py $HISTORY
