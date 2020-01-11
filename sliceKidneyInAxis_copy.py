import SimpleITK as sitk
import numpy as np
import argparse
import copy
import os
import sys
from functions import createParentPath, write_file
from cut import *
from pathlib import Path

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("filePath", help="/home/vmlab/Desktop/data/box/AABB/nonBlack/case_00000")
    parser.add_argument("savePath", help="$HOME/Desktop/data/slice/hist_0.0/case_00000")

    args = parser.parse_args()
    return args

def main(args):
    path = Path(args.filePath)

    leftImagePath = path / "image_left.nii.gz"
    rightImagePath = path / "image_right.nii.gz"

    leftImage = sitk.ReadImage(str(leftImagePath))
    rightImage = sitk.ReadImage(str(rightImagePath))

    leftImageArray = sitk.GetArrayFromImage(leftImage)
    rightImageArray = sitk.GetArrayFromImage(rightImage)

    leftLabelPath = path / "label_left.nii.gz"
    rightLabelPath = path / "label_right.nii.gz"

    leftLabel = sitk.ReadImage(str(leftLabelPath))
    rightLabel = sitk.ReadImage(str(rightLabelPath))

    leftLabelArray = sitk.GetArrayFromImage(leftLabel)
    rightLabelArray = sitk.GetArrayFromImage(rightLabel)

    length = leftImageArray.shape[2]

    imagePath = []
    labelPath = []
    for x in range(length):
        leftImageSlice = leftImageArray[:,:,x]
        rightImageSlice = rightImageArray[:,:,x]

        savePath = Path(args.savePath)
        leftImageSavePath = savePath / ("left/image_" + str(x).zfill(3) + ".mha")
        rightImageSavePath = savePath / ("right/image_" + str(x).zfill(3) + ".mha")

        createParentPath(leftImageSavePath)
        createParentPath(rightImageSavePath)

        leftLabelSlice = leftLabelArray[:,:,x]
        rightLabelSlice = rightLabelArray[:,:,x]

        leftLabelSavePath = savePath / ("left/label_" + str(x).zfill(3) + ".mha")
        rightLabelSavePath = savePath / ("right/label_" + str(x).zfill(3) + ".mha")
        
        sitk.WriteImage(sitk.GetImageFromArray(leftImageSlice), "test/test_{}.mha".format(x), True)
        save_image_256(leftImageSlice, leftImage, str(leftImageSavePath))
        save_image_256(rightImageSlice, rightImage, str(rightImageSavePath))
        save_image_256(leftLabelSlice, leftLabel, str(leftLabelSavePath), is_lab=True)
        save_image_256(rightLabelSlice, rightLabel, str(rightLabelSavePath), is_lab=True)

        imagePath.append(leftImageSavePath)
        imagePath.append(rightImageSavePath)
        labelPath.append(leftLabelSavePath)
        labelPath.append(rightLabelSavePath)

    textSavePath = savePath.parent / "path" / (path.name + ".txt")
    createParentPath(textSavePath)

    imagePath = sorted(imagePath)
    labelPath = sorted(labelPath)
    print(len(imagePath), len(labelPath))
    for x, y in zip(imagePath, labelPath):
        write_file(textSavePath, str(x) + "\t" + str(y))
            

        
if __name__ == '__main__':
    args = ParseArgs()
    main(args)
