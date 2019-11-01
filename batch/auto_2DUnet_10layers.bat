@echo off  
SETLOCAL ENABLEDELAYEDEXPANSION

cd %~dp0

::data path
set data= E:\kits19\data\case_00
set toolbin=C:\study\bin\imageproc\bin\imageproc.exe
::file name
set ct=\imaging.nii.gz
set cthist=\ct_hist25.mha
set label=\segmentation.nii.gz
set label_mask=\mask.mha



set number=0,1,2,3,4,5,6,7,8,9

for %%i in (%number%) do (
    set ctpath=%data%%%i%ct%
    set cthistpath=%data%%%i%cthist%
    set labpath=%data%%%i%label%
    set mask_path=%data%%%i%label_mask%

    buildUnet.py C:\Users\VMLAB\Desktop\2Dkidney\trainingList\margeTraining_%%i.txt 2DUnetModel_%%i.yml -t C:\Users\VMLAB\Desktop\2Dkidney\validationList\margeValidation_%%i.txt -b 15 -e 100 --bestfile C:\Users\VMLAB\Desktop\2Dkidney\weightFolder\best_%%i.hdf5 --history C:\Users\VMLAB\Desktop\2Dkidney\history\history_%%i.txt
    mail.py C:\Users\VMLAB\Desktop\2Dkidney\history\history_%%i.txt
)
