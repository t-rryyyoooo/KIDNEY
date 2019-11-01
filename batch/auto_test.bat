@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

cd %~dp0

::data path
set tra=C:\Users\VMLAB\Desktop\2Dkidney\trainingList\margeTraining_re_5_
set mod=C:\Users\VMLAB\Desktop\2Dkidney\modelFolder\2DUnetModel_re_5_
set val=C:\Users\VMLAB\Desktop\2Dkidney\validationList\margeValidation_re_5_
set wei=C:\Users\VMLAB\Desktop\2Dkidney\weightFolder\best_re_5_
set his=C:\Users\VMLAB\Desktop\2Dkidney\history\history_re_5_

set number=4

for %%i in (%number%) do (
    set training=%tra%%%i.txt
    set model=%mod%%%i.yml
    set validation=%val%%%i.txt
    set weight=%wei%%%i.hdf5
    set history=%his%%%i.txt

    python buildUnet.py test.txt testModel.yml -b 2 -e 2 -t ttest.txt --history history.txt
    python mail.py history.txt
)

