import SimpleITK as sitk
import numpy as np
import argparse
import copy
import os
import sys
from functions import createParentPath, write_file, Resampling
from cut import *
from pathlib import Path

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()

    #Change!!!!########################################################
    parser.add_argument("filePath", help="$HOME/Desktop/data/kits19/case_00000")
    parser.add_argument("savePath", help="$HOME/Desktop/data/slice/hist_0.0")

    ###################################################################

    args = parser.parse_args()
    return args

def main(args):
    path = Path(args.filePath)

    leftImagePath = path / "image_left.nii.gz"
    rightImagePath = path / "imge_right.nii.gz"
    leftLabelPath = path / "label_left.nii.gz"
    rightLabelPath = path / "label_right.nii.gz"

    textSavePath = path.parent / "path" / (path.name + ".txt")
    createParentPath(textSavePath)

    leftImage = sitk.ReadImage(str(leftImagePath))
    rightImage = sitk.ReadImage(str(rightImagePath))
    leftLabel = sitk.ReadImage(str(leftLabelPath))
    rightLabel = sitk.ReadImage(str(rightLabelPath))

    leftImageArray = sitk.GetArrayFromImage(leftImage)
    rightImageArray = sitk.GetArrayFromImage(rightImage)
    leftLabelArray = sitk.GetArrayFromImage(leftLabel)
    rightLabelArray = sitk.GetArrayFromImage(rightLabel)

    length = leftImageArray.shape[2]

    imagePath = []
    labelPath = []
    for x in range(length):
        leftImageSlice = leftImageArray[:,:,x]
        rightImageSlice = rightImageArray[:,:,x]
        leftLabelSlice = leftLabelArray[:,:,x]
        rightLabelSlice = rightLabelArray[:,:,x]

        savePath = Path(args.savePath)
        leftImageSavePath = savePath / "left/image_" + str(x).zfill(3) + ".mha"
        rightImageSavePath = savePath / "right/image_" + str(x).zfill(3) + ".mha"
        leftLabelSavePath = savePath / "left/label_" + str(x).zfill(3) + ".mha"
        rightLabelSavePath = savePath / "right/label_" + str(x).zfill(3) + ".mha"

        createParentPath(leftImageSavePath)

        save_image_256(leftImageSlice, leftImage, leftImageSavePath)
        save_image_256(rightImageSlice, rightImage, rightImageSavePath)
        save_image_256(leftLabelSlice, leftLabel, leftLabelSavePath, ia_lab=True)
        save_image_256(rightLabelSlice, rightLabel, rightLabelSavePath, is_lab=True)

        imagePath.append(leftImageSavePath)
        imagePath.append(rightImageSavePath)
        labelPath.append(leftLabelSavePath)
        labelPath.append(rightLabelSavePath)

    imagePath = sorted(imagePath)
    labelPath = sorted(labelPath)
    for x, y in zip(imagePath, labelPath):
        write_file(textSavePath, x + "\t" + y)
            
            write_file(str(OPT), str(OPL) + "\t" + str(OPI))

        
if __name__ == '__main__':
    args = ParseArgs()
    main(args)
