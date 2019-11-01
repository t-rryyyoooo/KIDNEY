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
set slice=E:\slice\image\case_00

::patch size
set psize=36x36x28

set numArr=000,001,002,003,004,005,006,007,008,009,010,011,012,013,014,015,016,017,018,019,020,021,022,023,024,025,026,027,028,029
set number=0,1,2,3,4,5,6,7,8,9

for %%i in (%numArr%) do (
    for %%r in (%number%) do(
        for %%f (%slice%%%i\%%r) do(
    set ctpath=%data%%%i%ct%
    set cthistpath=%data%%%i%cthist%
    set labpath=%data%%%i%label%
    set mask_path=%data%%%i%label_mask%

    ls

)
)
)
