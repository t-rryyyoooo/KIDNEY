@echo off  
SETLOCAL ENABLEDELAYEDEXPANSION

cd %~dp0

set slice=E:\slice\layers_1_interpolated\path\case_00
set save=E:\slice\layer_1_hist_org\segmentation\case_00


set numArr=000,001,002,003,004,006,007,008,009,010,011,012,013,014,015,016,017,018,019,020,021,022,023,024,025,026,027,028,029

set number=0,1,2,3,4

for %%i in (%numArr%) do (
   
    
    set ctlist=%slice%%%i.txt
    set model=C:\Users\VMLAB\Desktop\2Dkidney\modelFolder\2DUnetModel_OHistOrg.yml
    set weight=C:\Users\VMLAB\Desktop\2Dkidney\weightFolder\best_OHistOrg.hdf5
    set savepath=%save%%%i
   
    
    
    python segmentationUnet3ch.py !ctlist! !model! !weight! !savepath!
    



)
