@echo off  
SETLOCAL ENABLEDELAYEDEXPANSION

cd %~dp0
set data=E:\kits19\data\case_00
set save=E:\slice\test\till\\case_00
set weight=C:\Users\VMLAB\Desktop\secondKidney\weightFolder\best_OHistOrgRandomGraduallyTill0.25.hdf5
set model=C:\Users\VMLAB\Desktop\secondKidney\modelFolder\2DUnetModel_OHistOrgRandomGraduallyTill.yml

set image=\imaging.nii.gz
set label=\segmentation.nii.gz
set numArr=000,001,002,003,004,006,007,008,009,010,011,012,013,014,015,016,017,018,019,020,021,022,023,024,025,026,027,028,029
set num=000
set numRandom=173,002,068,133,155,114,090,105,112,175,183,208,029,065,157,162,141,062,031,156,189,135,020,077,000,009,198,036,


for %%i in (%numRandom%) do (

    set ctlist=%slice%%%i.txt
    set savepath=%save%%%i\label.mha
    set ct=%data%%%i%image%
    set lab=%data%%%i%label%
    

    python segmentationUnet3chAtOnce.py !lab! !ct! !model! !weight! !savepath! 0.25

    


)
)
