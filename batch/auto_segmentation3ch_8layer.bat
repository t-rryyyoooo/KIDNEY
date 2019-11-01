@echo off  
SETLOCAL ENABLEDELAYEDEXPANSION

cd %~dp0

set slice=E:\slice\layers_8_3ch\path\case_00
set save=E:\slice\layers_8_3ch\segmentation\case_00
set wei=C:\Users\VMLAB\Desktop\2Dkidney\weightFolder\best_8_
set mod=C:\Users\VMLAB\Desktop\2Dkidney\modelFolder\2DUnetModel_8_
set suffix=_3ch

set numArr=000,001,002,003,004,006,007,008,009,010,011,012,013,014,015,016,017,018,019,020,021,022,023,024,025,026,027,028,029
set numbers=0,1,2,3,4,5,6,7

for %%i in (%numArr%) do (
    for %%r in (%numbers%) do (

    set ctlist=%slice%%%i\%%r.txt
    set savepath=%save%%%i
    set weight=%wei%%%r%suffix%.hdf5
    set model=%mod%%%r%suffix%.yml
   
    
    
    python segmentationUnet3ch8layers.py !ctlist! !model! !weight! !savepath!
    

)

)
