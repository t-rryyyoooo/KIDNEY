#!/bin/bash

#Input
readonly TRAINING="$HOME/Desktop/data/textList/training_orignal.txt"
readonly VALIDATION="$HOME/Desktop/data/textList/validation_original.txt"
readonly WEIGHT="$HOME/Desktop/data/modelweight/best_6ch_3ch.hdf5"
readonly HISTORY="$HOME/Desktop/data/history/history_6ch_3ch.txt"

echo -n GPU_ID:
read id

echo $TRAINING
echo $VALIDATION
echo $WEIGHT
echo $HISTORY

#python3 buildUnet3chAugumentation.py ${TRAINING} ${WEIGHT} -t ${VALIDATION} --history ${HISTORY} -b 10 -e 40 -g $id

#python3 mail.py $HISTORY
