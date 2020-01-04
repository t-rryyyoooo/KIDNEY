from pathlib import Path
import argparse
import SimpleITK as sitk
import numpy as np
from clip3D import Resizing
from functions import saveImage, createParentPath, write_file

args = None

def parseArgs():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("filePath", help="$HOME/Desktop/data/box/case_00000")
    parser.add_argument("savePath", help="$HOME/Desktop/data/slice/6ch/original/case_00000")
    parser.add_argument("--suffix", default ="", help="transformed")
    args = parser.parse_args()
    return args

def saveSliceImage256(imgArray, img, savePath, interpolation):
    argMax = np.argmax(np.array(imgArray.shape))
    savePathList = []
    if argMax == 0:
        axisSize = imgArray.shape[0]
        dummyArray = np.zeros((axisSize, 256, 256))
        resizedImgArray = Resizing(imgArray, dummyArray, interpolation)
        
        for x in range(axisSize):
            saveSlicePath = savePath + str(x).zfill(3) + ".mha"
            savePathList.append(saveSlicePath)
            createParentPath(saveSlicePath)
            resizedImg = sitk.GetImageFromArray(resizedImgArray[x, :, :])
            
            direction = (0.0, 1.0, -1.0, 0.0)
            origin = img.GetOrigin()[:2]
            spacing = img.GetSpacing()[:2]
            
            resizedImg.SetDirection(direction)
            resizedImg.SetOrigin(origin)
            resizedImg.SetSpacing(spacing)
            
            sitk.WriteImage(resizedImg, saveSlicePath)

    elif argMax == 1:
        axisSize = imgArray.shape[1]
        dummyArray = np.zeros((256, axisSize, 256))

        resizedImgArray = Resizing(imgArray, dummyArray, interpolation)
        
        for x in range(axisSize):
            
            saveSlicePath = savePath + str(x).zfill(3) + ".mha"
            savePathList.append(saveSlicePath)
            createParentPath(saveSlicePath)
            resizedImg = sitk.GetImageFromArray(resizedImgArray[:, x, :])
            
            direction = (0.0, 1.0, -1.0, 0.0)
            origin = (img.GetOrigin()[0], img.GetOrigin()[2])
            spacing = (img.GetSpacing()[0], img.GetSpacing()[2])
            
            resizedImg.SetDirection(direction)
            resizedImg.SetOrigin(origin)
            resizedImg.SetSpacing(spacing)
            
            sitk.WriteImage(resizedImg, saveSlicePath)

            
    else:
        axisSize = imgArray.shape[2]
        dummyArray = np.zeros((256, 256, axisSize))

        resizedImgArray = Resizing(imgArray, dummyArray, interpolation)
        
        for x in range(axisSize):
            
            saveSlicePath = savePath + str(x).zfill(3) + ".mha"
            savePathList.append(saveSlicePath)
            createParentPath(saveSlicePath)
            resizedImg = sitk.GetImageFromArray(resizedImgArray[:, :, x])
            
            direction = (0.0, 1.0, -1.0, 0.0)
            origin = img.GetOrigin()[1:]
            spacing = img.GetSpacing()[1:]
            
            resizedImg.SetDirection(direction)
            resizedImg.SetOrigin(origin)
            resizedImg.SetSpacing(spacing)
            
            sitk.WriteImage(resizedImg, saveSlicePath)

    return savePathList


def main(args):
    leftLabelPath = args.filePath + "/label_left" + args.suffix + ".nii.gz"
    leftImagePath = args.filePath + "/image_left" + args.suffix + ".nii.gz"
    rightLabelPath = args.filePath + "/label_right" + args.suffix + ".nii.gz"
    rightImagePath = args.filePath + "/image_right" + args.suffix + ".nii.gz"



    leftLabel = sitk.ReadImage(leftLabelPath)
    leftLabelArray = sitk.GetArrayFromImage(leftLabel)

    leftImage = sitk.ReadImage(leftImagePath)
    leftImageArray = sitk.GetArrayFromImage(leftImage)

    rightLabel = sitk.ReadImage(rightLabelPath)
    rightLabelArray = sitk.GetArrayFromImage(rightLabel)

    rightImage = sitk.ReadImage(rightImagePath)
    rightImageArray = sitk.GetArrayFromImage(rightImage)
    
    saveLeftLabelPath = args.savePath + "/left/label_" 
    saveLeftImagePath = args.savePath +  "/left/image_" 
    saveRightLabelPath = args.savePath + "/right/label_" 
    saveRightImagePath = args.savePath + "/right/image_" 
    saveTextPath = Path(args.savePath).parent / "path" / (Path(args.savePath).name + ".txt")
    
    leftLabelPathList = saveSliceImage256(leftLabelArray, leftLabel, saveLeftLabelPath, "nearest")
    leftImagePathList = saveSliceImage256(leftImageArray, leftImage, saveLeftImagePath, "linear")
    rightLabelPathList = saveSliceImage256(rightLabelArray, rightLabel, saveRightLabelPath, "nearest")
    rightImagePathList = saveSliceImage256(rightImageArray, rightImage, saveRightImagePath, "linear")
   
    print(len(leftLabelPathList), len(leftImagePathList), len(rightLabelPathList), len(rightImagePathList))
    for ll, li in zip(leftLabelPathList, leftImagePathList):
        write_file(str(saveTextPath), ll + "\t" + li)
        
    for rl, ri in zip(rightLabelPathList, rightImagePathList):
        write_file(str(saveTextPath), rl + "\t" + ri)
    
    
if __name__=="__main__":
    args = parseArgs()
    main(args)
