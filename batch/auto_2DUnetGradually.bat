@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

cd %~dp0

::data path
set training=C:\Users\VMLAB\Desktop\secondKidney\trainingList\margeTraining_Gradually
set validation=C:\Users\VMLAB\Desktop\secondKidney\validationList\margeValidation_Gradually


set model=C:\Users\VMLAB\Desktop\secondKidney\modelFolder\2DUnetModel_OHistOrgRandomGraduallyTill.yml
set weight=C:\Users\VMLAB\Desktop\secondKidney\weightFolder\best_OHistOrgRandomGraduallyTill
set history=C:\Users\VMLAB\Desktop\secondKidney\history\history_OHistOrgRandomGraduallyTill.txt

set latest=log/latestweights.hdf5
set rate[0]=0.0
set rate[1]=0.05
set rate[2]=0.10
set rate[3]=0.15
set rate[4]=0.20
set rate[5]=0.25


set startEpoch[0]=0
set startEpoch[1]=17
set startEpoch[2]=33
set startEpoch[3]=49
set startEpoch[4]=65
set startEpoch[5]=81


set endEpoch[0]=16
set endEpoch[1]=32
set endEpoch[2]=48
set endEpoch[3]=64
set endEpoch[4]=80
set endEpoch[5]=100


for %%i in (0,1,2,3,4,5) do (
    
    set trainingList=%training%!rate[%%i]!.txt
    set validationList=%validation%!rate[%%i]!.txt
    set weightfiles=%weight%!rate[%%i]!.hdf5
    echo !trainingList!
    echo !validationList!
    echo !startEpoch[%%i]!
    echo !endEpoch[%%i]!
    echo !weightfiles!

    if %%i==0 (

    python buildUnet3chAugmentation.py !trainingList! %model% -t !validationList!  -b 15 --initialepoch !startEpoch[%%i]! -e !endEpoch[%%i]! --bestfile !weightfiles! --history %history% 

  ) else (

    python buildUnet3chAugmentation.py !trainingList! %model% -t !validationList!  -b 15 --initialepoch !startEpoch[%%i]! -e !endEpoch[%%i]! --bestfile !weightfiles! --weightfile %latest%  --history %history% 
)
    

)
python mail.py %history%
