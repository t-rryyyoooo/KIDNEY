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
    savePath = Path(args.savePath)
    saveSize = (256, 256)

    imagePathList = []
    labelPathList = []

    for d in ["left", "right"]:
        imagePath = path / ("image_" + d + ".nii.gz")
        labelPath = path / ("label_" + d + ".nii.gz")

        image = sitk.ReadImage(str(imagePath))
        label = sitk.ReadImage(str(labelPath))

        dummyPath = savePath / d / "dummy.mha"
        createParentPath(dummyPath)

        length = image.GetSize()[0]
        for x in range(length):
            imageSavePath = savePath / d / ("image_" + str(x).zfill(3) + ".mha")
            labelSavePath = savePath / d / ("label_" + str(x).zfill(3) + ".mha")


            imageSlice = ResamplingInAxis(image, x, saveSize)
            labelSlice = ResamplingInAxis(label, x, saveSize, is_label=True)

            sitk.WriteImage(imageSlice, str(imageSavePath), True)
            sitk.WriteImage(labelSlice, str(labelSavePath), True)


            imagePathList.append(str(imageSavePath))
            labelPathList.append(str(labelSavePath))


    textSavePath = savePath.parent / "path" / (path.name + ".txt")
    createParentPath(textSavePath)

    imagePathList = sorted(imagePathList)
    labelPathList = sorted(labelPathList)
    for x, y in zip(imagePathList, labelPathList):
        write_file(textSavePath, str(y) + "\t" + str(x))
            

        
if __name__ == '__main__':
    args = ParseArgs()
    main(args)
