#!/bin/bash

#Input
readonly TRAINING="C:\Users\VMLAB\Desktop\secondKidney\trainingList\margeTraining_"
readonly VALIDATION="C:\Users\VMLAB\Desktop\secondKidney\validationList\margeValidation_"
readonly MODEL="C:\Users\VMLAB\Desktop\secondKidney\modelFolder\2DUnetModel_"
readonly WEIGHT="C:\Users\VMLAB\Desktop\secondKidney\weightFolder\best_"
readonly HISTORY="C:\Users\VMLAB\Desktop\secondKidney\history\history_"

readonly ALPHA=(0.55 0.60 0.65 0.70 0.80 0.85 0.90 0.95 1.0)

for alpha in ${ALPHA[@]}
do

    training=${TRAINING}${alpha}.txt
    validation=${VALIDATION}${alpha}.txt
    weight=${WEIGHT}${alpha}.hdf5
    model=${MODEL}${alpha}.yml
    histories=${HISTORY}${alpha}.txt

    echo training
    echo validation
    echo weight
    echo model 
    echo histories

    python __version__
    python buildUnet3chAugmentation.py ${training} ${model} -t ${validation} --bestfile !${weight} --history ${histories} -b 15  -e 40
    python mail.py %history%

done