@echo off  
SETLOCAL ENABLEDELAYEDEXPANSION

cd %~dp0
set data=E:\kits19\data\case_00
::set save=E:\slice\test\0.50\case_00
::set weight=C:\Users\VMLAB\Desktop\secondKidney\weightFolder\best_OHistOrgRandomGradually0.5.hdf5
set image=\imaging.nii.gz
set label=\segmentation.nii.gz
set numArr=000,001,002,003,004,006,007,008,009,010,011,012,013,014,015,016,017,018,019,020,021,022,023,024,025,026,027,028,029

set numRandom=173,002,068,133,155,114,090,105,112,175,183,208,029,065,157,162,141,062,031,156,189,135,020,077,000,009,198,036,

set rate=0.50,0.55,0.60,0.65,0.70
for %%r in (%rate%)do (
    for %%i in (%numRandom%) do (

        set weight=C:\Users\VMLAB\Desktop\KIDNEY\weightFolder\best_re%%r.hdf5
        set model=C:\Users\VMLAB\Desktop\KIDNEY\modelFolder\2DUnetModel_re%%r.yml
        set save=E:\slice\layers_1_hist_org_%%r\segmentation\case_00
        set savepath=!save!%%i\label.mha
        set ct=%data%%%i%image%
        set lab=%data%%%i%label%
        
        echo !weight!
        echo !model!
        echo !savepath!

        python ../segmentationUnet3chAtOnce.py !lab! !ct! !model! !weight! !savepath! %%r

    


    )
)
