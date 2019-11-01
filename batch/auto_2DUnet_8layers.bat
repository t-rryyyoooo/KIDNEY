@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

cd %~dp0

::data path
set tra=C:\Users\VMLAB\Desktop\2Dkidney\trainingList\margeTraining_8_
set mod=C:\Users\VMLAB\Desktop\2Dkidney\modelFolder\2DUnetModel_8_
set val=C:\Users\VMLAB\Desktop\2Dkidney\validationList\margeValidation_8_
set wei=C:\Users\VMLAB\Desktop\2Dkidney\weightFolder\best_8_
set his=C:\Users\VMLAB\Desktop\2Dkidney\history\history_8_
set suffix=_3ch
set number=0,1,2,3,4,5,6,7

for %%i in (%number%) do (
    set training=%tra%%%i%suffix%.txt
    set model=%mod%%%i%suffix%.yml
    set validation=%val%%%i%suffix%.txt
    set weight=%wei%%%i%suffix%.hdf5
    set history=%his%%%i%suffix%.txt

    python buildUnet3ch8layers.py !training! !model! -t !validation!  -b 15 -e 100 --bestfile !weight! --history !history!
    python mail.py !history!
)