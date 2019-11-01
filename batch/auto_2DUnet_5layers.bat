@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

cd %~dp0

::data path
set tra=C:\Users\VMLAB\Desktop\2Dkidney\trainingList\margeTraining_re_5_
set mod=C:\Users\VMLAB\Desktop\2Dkidney\modelFolder\2DUnetModel_re3ch_5_
set val=C:\Users\VMLAB\Desktop\2Dkidney\validationList\margeValidation_re_5_
set wei=C:\Users\VMLAB\Desktop\2Dkidney\weightFolder\best_re3ch_5_
set his=C:\Users\VMLAB\Desktop\2Dkidney\history\history_re3ch_5_

set number=0

for %%i in (%number%) do (
    set training=%tra%%%i.txt
    set model=%mod%%%i.yml
    set validation=%val%%%i.txt
    set weight=%wei%%%i.hdf5
    set history=%his%%%i.txt

    python buildUnet3ch.py !training! !model! -t !validation!  -b 17 -e 1 --bestfile !weight! --history !history!
    python mail.py !history!
)

