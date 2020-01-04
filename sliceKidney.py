from pathlib import Path
import argparse
import SimpleITK as sitk
import numpy as np
from clip3D import Resizing
from functions import saveImage, createParentPath, write_file
from cut import saveSliceImage256

args = None

def parseArgs():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("filePath", help="$HOME/Desktop/data/box/case_00000")
    parser.add_argument("savePath", help="$HOME/Desktop/data/slice/6ch/original/case_00000")
    parser.add_argument("--suffix", default ="", help="transformed")
    args = parser.parse_args()
    return args


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
