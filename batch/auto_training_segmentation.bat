@echo off  
SETLOCAL ENABLEDELAYEDEXPANSION

cd %~dp0
set training=C:\Users\VMLAB\Desktop\secondKidney\trainingList\margeTraining_pandas3.txt
set validation=C:\Users\VMLAB\Desktop\secondKidney\validationList\margeValidation_pandas3.txt
set weight=C:\Users\VMLAB\Desktop\secondKidney\weightFolder\best_pandas3.hdf5
set model=C:\Users\VMLAB\Desktop\secondKidney\modelFolder\2DUnetModel_pandas3.yml
set save=E:\slice\layers_1_hist_org_0.25\segmentation\case_00
set history=C:\Users\VMLAB\Desktop\secondKidney\history\history_pandas3.txt

set extractSave=E:/showCT/case_00

set data=E:\kits19\data\case_00
set image=\imaging.nii.gz
set label=\segmentation.nii.gz

set numArr=000,001,002,003,004,006,007,008,009,010,011,012,013,014,015,016,017,018,019,020,021,022,023,024,025,026,027,028,029
set num=000
set numRandom=006,052,075,115,065,022,061,181,012,107,028,079,031,086,044,146,027,164,003,090,150,084,183,161,005,193,154,100,036,001
set numRandomPandas2=167,095,117,082,022,174,176,012,008,178,025,171,146,019,045,048,039,001,002,133,209,060,125,155,046,143,053,020,152,102
set numRandomPandas3=059,096,169,075,051,175,010,157,186,108,042,166,027,017,201,133,031,076,002,032,150,060,069,155,180,143,086,151,118,013,168

python buildUnet3chAugmentation.py %training% %model% -t %validation%  -b 15 -e 100 --bestfile %weight% --history %history% 
python mail.py %history%

for %%i in (%numRandomPandas3%) do (

    set savepath=%save%%%i\label.mha
    set ct=%data%%%i%image%
    set lab=%data%%%i%label%
    set extractSavePath=%extractSave%%%i
    
    
    python segmentationUnet3chAtOnce.py !lab! !ct! !model! !weight! !savepath! 0.25

)
python mail.py %history%
