@echo off  
SETLOCAL ENABLEDELAYEDEXPANSION

cd %~dp0

set slice=E:\slice\layers_1\path\case_00
set save=E:\slice\layers_1\segmentation_Old\case_00
set weight=C:\Users\VMLAB\Desktop\2Dkidney\weightFolder\best_OlogDice.hdf5
set model=C:\Users\VMLAB\Desktop\2Dkidney\modelFolder\2DUnetModel_OlogDice.yml


set numArr=000,001,002,003,004,006,007,008,009,010,011,012,013,014,015,016,017,018,019,020,021,022,023,024,025,026,027,028,029
set num=000


for %%i in (%num%) do (

    set ctlist=%slice%%%i.txt
    set savepath=%save%%%i
   
    
    
    python segmentationUnet3ch.py !ctlist! !model! !weight! !savepath!
    



)
