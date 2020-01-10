import SimpleITK as sitk
from clip3D import searchBound, adjustDiff, searchBoundAndMakeIndex
import numpy as np
import argparse
from pathlib import Path
from functions import saveImage, createParentPath

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagePath", help="/vmlab/Desktop/data/kit19/case_00000")
    parser.add_argument("savePath", help="/vmlab/data/box/case_00000")
    parser.add_argument("--nonBlack", help="Change anything other than kidneys to black.", action="store_true")
    parser.add_argument("--expansion", help="0", default = 0, type=int)
    parser.add_argument("--prefix", help="resampled_", default="")
    args = parser.parse_args()

    return args
def main(args):

    path = Path(args.imagePath)
    imagePath = path / (args.prefix + 'imaging.nii.gz')
    labelPath = path / (args.prefix + 'segmentation.nii.gz')
    image = sitk.ReadImage(str(imagePath))
    label = sitk.ReadImage(str(labelPath))
    print(label.GetSize())


    imageArray = sitk.GetArrayFromImage(image)
    labelArray = sitk.GetArrayFromImage(label)

    startIdx, endIdx = searchBound(labelArray, 'sagittal')
    startIdx, endIdx = adjustDiff(startIdx, endIdx, labelArray.shape[0])

    startIdx -= args.expansion
    endIdx += args.expansion

    clipLabelArray = {"left" : labelArray[startIdx[0] : endIdx[0] + 1,...], 
                      "right" : labelArray[startIdx[1] : endIdx[1] + 1,...]}
    clipImageArray = {"left" : imageArray[startIdx[0] : endIdx[0] + 1,...], 
                      "right" : imageArray[startIdx[1] : endIdx[1] + 1,...]}

    print("Clipping images in sagittal direction...")
    print("Left Kidney clipped size : {}".format(clipLabelArray["left"].shape))
    print("Right Kidney clipped size : {}".format(clipLabelArray["right"].shape))

    startIdx, endIdx = searchBoundAndMakeIndex(clipLabelArray, "axial")
    startIdx, endIdx = adjustDiff(startIdx, endIdx, labelArray.shape[2])

    startIdx -= args.expansion
    endIdx += args.expansion

    clipLabelArray = {"left" : clipLabelArray["left"][..., startIdx[0] : endIdx[0] + 1], 
                      "right" : clipLabelArray["right"][..., startIdx[1] : endIdx[1] + 1]}
    clipImageArray = {"left" : clipImageArray["left"][..., startIdx[0] : endIdx[0] + 1], 
                      "right" : clipImageArray["right"][..., startIdx[1] : endIdx[1] + 1]}

    print("Clipping images in axial direction...")
    print("Left Kidney clipped size : {}".format(clipLabelArray["left"].shape))
    print("Right Kidney clipped size : {}".format(clipLabelArray["right"].shape))

    startIdx, endIdx = searchBoundAndMakeIndex(clipLabelArray, "coronal")
    startIdx, endIdx = adjustDiff(startIdx, endIdx, labelArray.shape[1])
    startIdx -= args.expansion
    endIdx += args.expansion
    clipLabelArray = {"left" : clipLabelArray["left"][:, startIdx[0] : endIdx[0] + 1, :], 
                      "right" : clipLabelArray["right"][:, startIdx[1] : endIdx[1] + 1, :]}
    clipImageArray = {"left" : clipImageArray["left"][:, startIdx[0] : endIdx[0] + 1, :], 
                      "right" : clipImageArray["right"][:, startIdx[1] : endIdx[1] + 1, :]}

    print("Clipping images in coronal direction...")
    print("Left Kidney clipped size : {}".format(clipLabelArray["left"].shape))
    print("Right Kidney clipped size : {}".format(clipLabelArray["right"].shape))

    clipLabelArray["right"] = clipLabelArray["right"][::-1,...]
    clipImageArray["right"] = clipImageArray["right"][::-1,...]

    if not args.nonBlack:
        leftIdx = np.where(clipLabelArray["left"] > 0, True, False)
        clipImageArray["left"] = np.where(leftIdx, clipImageArray["left"], -1024)
        rightIdx= np.where(clipLabelArray["right"] > 0, True, False)
        clipImageArray["right"] = np.where(rightIdx, clipImageArray["right"], -1024)
        

    clipLabelArray["left"] = sitk.GetImageFromArray(clipLabelArray["left"])
    clipLabelArray["right"] = sitk.GetImageFromArray(clipLabelArray["right"])
    clipImageArray["left"] = sitk.GetImageFromArray(clipImageArray["left"])
    clipImageArray["right"] = sitk.GetImageFromArray(clipImageArray["right"])

    savePath = Path(args.savePath)
    leftLabelPath = savePath / "label_left.nii.gz"
    leftImagePath = savePath / "image_left.nii.gz"
    rightLabelPath = savePath / "label_right.nii.gz"
    rightImagePath = savePath / "image_right.nii.gz"

    createParentPath(leftLabelPath)
    saveImage(clipLabelArray["left"], label, str(leftLabelPath))
    saveImage(clipLabelArray["right"], label, str(rightLabelPath))
    saveImage(clipImageArray["left"], label, str(leftImagePath))
    saveImage(clipImageArray["right"], label, str(rightImagePath))

if __name__=="__main__":
    args = ParseArgs()
    main(args)
