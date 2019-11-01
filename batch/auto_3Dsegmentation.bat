@echo off  
SETLOCAL ENABLEDELAYEDEXPANSION

cd %~dp0

set ct=\image.mha
set result=\result.mha
set sliceTo3D=E:\slice\sep3D\case_00
set save=E:\slice\segmentation\case_00
::patch size
set psize=36x36x28

set numArr=000,001,002,003,004,006,007,008,009,010,011,012,013,014,015,016,017,018,019,020,021,022,023,024,025,026,027,028,029
set number=0,1,2,3,4,5,6,7,8,9

for %%i in (%numArr%) do (
    for %%r in (%number%) do (

    set ctpath=%sliceTo3D%%%i\%%r%ct%
    set savepath=%save%%%i\%%r%result%
    
    
    segmentationUnet.py !ctpath! C:\Users\VMLAB\Desktop\2Dkidney\2DUnetModel_%%r.yml C:\Users\VMLAB\Desktop\2Dkidney\weightFolder\best_%%r.hdf5 !savepath!
    

)
)
