@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

cd %~dp0

::data path
set training=C:\Users\VMLAB\Desktop\KIDNEY\trainingList\margeTraining_
set validation=C:\Users\VMLAB\Desktop\KIDNEY\validationList\margeValidation_


set model=C:\Users\VMLAB\Desktop\KIDNEY\modelFolder\2DUnetModel_re
set weight=C:\Users\VMLAB\Desktop\KIDNEY\weightFolder\best_re
set history=C:\Users\VMLAB\Desktop\KIDNEY\history\history_re

set rate[0]=0.0
set rate[1]=0.05
set rate[2]=0.10
set rate[3]=0.15
set rate[4]=0.20
set rate[5]=0.25
set rate[6]=0.30
set rate[7]=0.35
set rate[8]=0.40
set rate[9]=0.45


for %%i in (0,1,2,3,4,5,6,7,8,9) do (
    
    set trainingList=%training%!rate[%%i]!.txt
    set validationList=%validation%!rate[%%i]!.txt
    set weightfiles=%weight%!rate[%%i]!.hdf5
    set histories=%history%!rate[%%i]!.txt
    set models=%model%!rate[%%i]!.yml

    echo !trainingList!
    echo !validationList!
    echo !weightfiles!
    echo !models!
    echo !histories!

   

    python ../buildUnet3chAugmentation.py !trainingList! !models! -t !validationList!  -b 15  -e 40 --bestfile !weightfiles! --history !histories! 
    python ../mail.py %history%
)

