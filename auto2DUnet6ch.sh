#!/bin/bash

#Input
readonly TRAINING="$HOME/Desktop/data/textList/training__"
readonly VALIDATION="$HOME/Desktop/data/textList/validation_"
readonly WEIGHT="$HOME/Desktop/data/modelweight/best_6ch_6ch.hdf5"
readonly HISTORY="$HOME/Desktop/data/history/history_6ch_6ch.txt"

echo -n GPU_ID:
read id

training1="${TRAINING}original.txt"
training2="${TRAINING}transform.txt"
validation1="${VALIDATION}original.txt"
validation2="${VALIFATION}transform.txt"

echo $training1
echo $training2
echo $validation1
echo $validation2
echo $WEIGHT
echo $HISTORY

#python3 buildUnet6chAugumentation.py ${training1} ${training2} ${WEIGHT} -t ${validation1} ${validation2} --history ${HISTORY} -b 10 -e 40 -g $id

#python3 mail.py $HISTORY
