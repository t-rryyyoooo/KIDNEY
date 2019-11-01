@echo off  
SETLOCAL ENABLEDELAYEDEXPANSION

cd %~dp0

set slice=E:\slice\layers_5\image\case_00
set save=E:\slice\onlyCanSliceOrg\segmentation\case_00


set numArr=000,001,002,003,004,006,007,008,009,010,011,012,013,014,015,016,017,018,019,020,021,022,023,024,025,026,027,028,029

set number=0,1,2,3,4

for %%i in (%numArr%) do (
    for %%r in (%number%) do (
        for %%f in (%slice%%%i\%%r\*) do (
    set model=C:\Users\VMLAB\Desktop\2Dkidney\modelFolder\2DUnetModel_ocs.yml
    set weight=C:\Users\VMLAB\Desktop\2Dkidney\weightFolder\best_ocs.hdf5
    set savepath=%save%%%i\%%r
    set p=%%f
    
    
    python segmentationUnet.py %%f !model! !weight! !savepath!\!p:~47!
    

)
)
)
