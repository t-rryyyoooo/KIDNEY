@echo off  
SETLOCAL ENABLEDELAYEDEXPANSION
::data path
set data=C:\study\vessel\Data\ID
set toolbin=C:\study\bin\imageproc\bin\imageproc.exe
::file name
set ct=\CT.mha
set label=\label.mha
set cthist=\hist_all_cut.mha
set newlabel=\new\label.mha
set label1=\label_HepaticVein.mha
set label2=\label_IVC.mha
set label3=\label_Liver.mha
set label4=\label_PortalVein.mha
set label5=\label_vessel.mha
set nl3=\new\new_PortalVein.mha
set nl4=\new\new_HepaticVein.mha
set nv=\new\label_vessel109.mha
set label_result1=\
set label_result2=\label_result_t.mha
set label_mask=\label_WholeLiver.mha
set weight1=C:\study\vessel\2DUnet\loghist100e\latestweights.hdf5
set weight2=C:\study\vessel\3DUnet\cross\006-009\trans_log\latestweights.hdf5

::patch size
set psize=36x36x28


:: NO 15 and 25
set numArr=006,007,008,009,010,011,012,013,014,016,017,018,019,020,021,022,023,024,026,027

for %%i in (%numArr%) do (
    set ctpath=%data%%%i%ct%
    set cthistpath=%data%%%i%cthist%
    set labpath=%data%%%i%label%
    set newlabpath=%data%%%i%newlabel%
    set l1path=%data%%%i%label1%
    set l2path=%data%%%i%label2%
    set l3path=%data%%%i%label3%
    set l4path=%data%%%i%label4%
    set l5path=%data%%%i%label5%
    set nl3path=%data%%%i%nl3%
    set nl4path=%data%%%i%nl4%
    set nvpath=%data%%%i%nv%
    set result1_path=.\loghist100e\label_2D_result%%i.mha

    set mask_path=%data%%%i%label_mask%
    set pinfo=.\info_newlabel\ID%%iinfo.txt
    
    segmentationUnet.py !cthistpath! test2D.hdf5 %weight1% !result1_path!
    %toolbin% load:!nvpath! similarity:!result1_path!,dice >>.\loghist100e\result.txt
    
)
    ::ExtractPatchImages.exe !pinfo! --input !l1path! !l4path! --outdir patch\vlabel%%i --patchlist .\list\vlID%%i.txt -t 8 --compose --withBG
    ::%toolbin% load:!l1path! add:!l4path! save:!l5path!
::    ExtractPatchInfo.exe !ctpath! !labpath! 36x36x28 !pinfo!
::    ExtractPatchImages.exe !pinfo! --input !ctpath! --outdir patch\origin%%i --patchlist .\list\oID%%i.txt -t 8

::    ExtractPatchImages.exe !pinfo! --input !l1path! !l2path! !l3path! !l4path! --outdir patch\label%%i --patchlist .\list\lID%%i.txt -t 8 --compose --withBG
::set numArr=006,007,008,009,010,011,012,013,014,016,017,018,019,020,021,022,023,024,026,027

::    segmentation3DUnet.py !ctpath! C:\study\vessel\3DUnet\onetype.yml %weight1% !result1_path! --mask !mask_path!
::    segmentation3DUnet.py !ctpath! C:\study\vessel\3DUnet\onetype.yml %weight2% !result2_path! --mask !mask_path!
::    %toolbin% load:!l5path! similarity:!result1_path!,dice >>result.txt
::    %toolbin% load:!l5path! similarity:!result2_path!,dice >>result2.txt
::  C:\study\vessel\3DUnet\ExtractPatchImages.exe C:\study\vessel\test\test\testinfo.txt --input C:\study\vessel\test\label.mha --outdir pathtest\labletest --patchlist .\listtest\list.txt -t 8 --compose --withBG
::    %toolbin% load:!newlabpath! divide:3 save:!nl3path!
::    %toolbin% load:!newlabpath! divide:4 save:!nl4path!
    ::ExtractPatchImages.exe !pinfo! --input !ctpath! --outdir newpatch\origin%%i --patchlist .\list\new2\lID%%i.txt -t 8
    ::    mkdir C:\study\vessel\2DUnet\Data\ID%%i
    ::extractSlices.py !ctpath! !nl3path! --outfilespec C:\study\vessel\2DUnet\Data\ID%%i\image{no:02d}A{slice:03d}.mha -o C:\study\vessel\2DUnet\Data\list\list%%i.txt
    ::segmentationUnet.py !ctpath! test2D.hdf5 %weight1% !result1_path!
    ::    segmentationUnet.py !ctpath! test2D.hdf5 %weight1% !result1_path!
    ::%toolbin% load:!nvpath! similarity:!result1_path!,dice >>result2d.txt
::extractSlices.py !cthistpath! !nvpath! --outfilespec C:\study\vessel\2DUnet\datahist\ID%%i\image{no:02d}A{slice:03d}.mha -o C:\study\vessel\2DUnet\datahist\list\list%%i.txt
