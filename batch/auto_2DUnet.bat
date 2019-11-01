@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

cd %~dp0

::data path
set training=C:\Users\VMLAB\Desktop\secondKidney\trainingList\margeTraining_OHistOrgRandom0.25.txt
set model=C:\Users\VMLAB\Desktop\secondKidney\modelFolder\2DUnetModel_OHistOrgRandom0.25.yml
set validation=C:\Users\VMLAB\Desktop\secondKidney\validationList\margeValidation_OHistOrgRandom0.25.txt
set weight=C:\Users\VMLAB\Desktop\secondKidney\weightFolder\best_OHistOrgRandom0.25.hdf5
set history=C:\Users\VMLAB\Desktop\secondKidney\history\history_OHistOrgRandom0.25.txt

    python buildUnet3chAugmentation.py %training% %model% -t %validation%  -b 15 -e 100 --bestfile %weight% --history %history% 
    python mail.py %history%


