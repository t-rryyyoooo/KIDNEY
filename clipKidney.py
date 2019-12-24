import SimpleITK as sitk
import numpy as np
from pyobb.obb import OBB
from functions import saveImage, printchk, createParentPath
from clip3D import searchBound, getSortedDistance, makeCompleteMatrix, determineSlide, determineClipSize, makeRefCoords, transformImageArray, reverseImage, Resizing
import argparse
from pathlib import Path

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagePath", help="/vmlab/Desktop/data/kit19/case_00000")
    parser.add_argument("savePath", help="/vmlab/data/box/case_00000")
    parser.add_argument("--log", help="To write patients failed to clip", default="log")
    parser.add_argument("--extension", help="nii.gz", default='nii.gz')

    args = parser.parse_args()

    return args

def main(args):
    # Read image
    labelPath = Path(args.imagePath) / "segmentation.nii.gz"
    imagePath = Path(args.imagePath) / "imaging.nii.gz"

    img = sitk.ReadImage(str(imagePath))
    imageArray = sitk.GetArrayFromImage(img)
    label = sitk.ReadImage(str(labelPath))
    labelArray = sitk.GetArrayFromImage(label)

    # Find a border dividing the kidney into a separate space 
    startIdx, endIdx = searchBound(labelArray, 'sagittal')
    print("startIndex : ", startIdx)
    print("endIndex : ", endIdx)

        # Divide the kidney into leftArray and rightArray
    leftArray = labelArray[ : startIdx[1] - 1, :, :]
    rightArray = labelArray[endIdx[0] + 1 : , :, :]
    print('leftArray shapes : ', leftArray.shape)
    print('rightArray shapes : ', rightArray.shape)

    leftImgArray = imageArray[ : startIdx[1] - 1,:,:]
    rightImgArray = imageArray[endIdx[0] + 1 : , :, :]
   
    arrayDict = {"left" : [leftArray, leftImgArray], "right" : [rightArray, rightImgArray]}

    for xxx in ["left", "right"]:
        # Input array
        inputLabelArray = arrayDict[xxx][0]
        inputImageArray = arrayDict[xxx][1]

        # Find kidney region
        idx = np.where(inputLabelArray > 0)

        # Preprocessing for OBB
        vertics = np.stack([*idx], axis=-1)

        # Implement OBB
        obb = OBB.build_from_points(vertics)

        # Minimum vertics for defining bounding box per vertex
        index = (((0, 1), (0, 3), (0, 5)),
                 ((1, 0), (1, 2), (1, 4)),
                 ((2, 1), (2, 3), (2, 7)),
                 ((3, 0), (3, 2), (3, 6)),
                 ((4, 1), (4, 5), (4, 7)),
                 ((5, 0), (5, 4), (5, 6)),
                 ((6, 3), (6, 5), (6, 7)), 
                 ((7, 2,), (7, 4), (7, 6)))

        # Find the closest vertex to origin
        minVertex = sorted([(x[0]**2 + x[1]**2 + x[2]**2, i) for i, x in enumerate(obb.points)])[0][1]
        print("minVertex : ", minVertex)

        # Calucualte the length of each side of bouding box and output in decending order
        points = getSortedDistance(obb.points, index[minVertex])

        #Slide to locate point[0][0] into (0,0,0)
        #points = points - points[0][0]

        rotationMatrix, rotatedBoundingVertics = makeCompleteMatrix(points)
        print("rotatedBoundingVertics")
        print(rotatedBoundingVertics.astype(int))

        #Measure for rotated coordinates are out of range of inputArray size
        slide, affineMatrix = determineSlide(rotatedBoundingVertics, rotationMatrix, inputLabelArray)

        # Slide boundingVertics
        print("slide : ", slide)
        rotatedAndSlidedBoundingVertics = rotatedBoundingVertics + slide
        print("rotatedAndSlidedBoundingVertics")
        print(rotatedAndSlidedBoundingVertics.astype(int))


        origin, clipSize = determineClipSize(rotatedAndSlidedBoundingVertics, inputLabelArray.shape)
        print("origin : ", origin)
        print("clipSize : ", clipSize)
        
        # For affine transformation, make inverse matrix
        invAffine = np.linalg.inv(affineMatrix)
        refCoords = makeRefCoords(inputLabelArray, invAffine)
        print("redCoords shape : ", refCoords.shape)
        
        print("Rotating image...")
        rotatedLabelArray = transformImageArray(inputLabelArray, refCoords, "nearest")
        rotatedImageArray = transformImageArray(inputImageArray, refCoords, "linear")

        rotatedLabelArray = rotatedLabelArray[origin[0] - 1 : clipSize[0] + 1,
                                              origin[1] - 1 : clipSize[1] + 1,
                                              origin[2] - 1 : clipSize[2] + 1]

        rotatedImageArray = rotatedImageArray[origin[0] - 1 : clipSize[0] + 1,
                                              origin[1] - 1 : clipSize[1] + 1,
                                              origin[2] - 1 : clipSize[2] + 1]



        pre = np.where(inputLabelArray > 0, True, False).sum()
        post = np.where(rotatedLabelArray > 0, True, False).sum()

        log = Path(args.log) / "failList.txt"
        saveLabelPath = Path(args.savePath) / ("label_" + xxx + "." + args.extension)
        saveImagePath = Path(args.savePath) / ("image_" + xxx + "." + args.extension)

        createParentPath(saveLabelPath)
        
        # reverse right image
        if xxx == "right":
            print("It is the right kidney which should be reversed.")
            rotatedLabelArray = reverseImage(rotatedLabelArray)
            rotatedImageArray = reverseImage(rotatedImageArray)

        if 0.99 < post / pre < 1.1:
            print("Succeeded in clipping.")
            rotatedLabel = sitk.GetImageFromArray(rotatedLabelArray)
            rotatedImage = sitk.GetImageFromArray(rotatedImageArray)
            
            saveImage(rotatedLabel, label, str(saveLabelPath))
            saveImage(rotatedImage, img, str(saveImagePath))


        else:
            print("Failed to clip.")
            print("Writing failed patient to {}".format(str(log)))
            print("Done")
    
    # Match one kidney shape with the other one.
    if (Path(args.savePath) / ("label_right." + args.extension)).exists() and (Path(args.savePath) / ("label_left." + args.extension)).exists():
        rightLabelPath = Path(args.savePath) / ("label_right." + args.extension)
        rightImagePath = Path(args.savePath) / ("image_right." + args.extension)
        leftLabelPath = Path(args.savePath) / ("label_left." + args.extension)
        leftImagePath = Path(args.savePath) / ("image_left." + args.extension)
        
        rightLabel = sitk.ReadImage(str(rightLabelPath))
        rightImage = sitk.ReadImage(str(rightImagePath))
        leftLabel = sitk.ReadImage(str(leftLabelPath))
        leftImage = sitk.ReadImage(str(leftImagePath))
        
        rightLabelArray  = sitk.GetArrayFromImage(rightLabel)
        rightImageArray = sitk.GetArrayFromImage(rightImage)
        leftLabelArray = sitk.GetArrayFromImage(leftLabel)
        leftImageArray = sitk.GetArrayFromImage(leftImage)
        
        
        rightLabelTransformedArray = Resizing(rightLabelArray, leftLabelArray,"nearest")
        rightImageTransformedArray = Resizing(rightImageArray, leftImageArray, "linear")
        leftLabelTransformedArray = Resizing(leftLabelArray, rightLabelArray,"nearest")
        leftImageTransformedArray = Resizing(leftImageArray, rightImageArray, "linear")
        
        saveRightLabelTransformedPath = Path(args.savePath) / ("label_right_transformed." + args.extension)
        saveRightImageTransformedPath = Path(args.savePath) / ("image_right_transformed." + args.extension)
        saveLeftLabelTransformedPath = Path(args.savePath) / ("label_left_transformed." + args.extension)
        saveLeftImageTransformedPath = Path(args.savePath) / ("image_left_transformed." + args.extension)
        
        rightLabelTransformed = sitk.GetImageFromArray(rightLabelTransformedArray)
        rightImageTransformed = sitk.GetImageFromArray(rightImageTransformedArray)
        leftLabelTransformed = sitk.GetImageFromArray(leftLabelTransformedArray)
        leftImageTransformed = sitk.GetImageFromArray(leftImageTransformedArray)

        saveImage(rightLabelTransformed, rightLabel, str(saveRightLabelTransformedPath))
        saveImage(rightImageTransformed, rightImage, str(saveRightImageTransformedPath))
        saveImage(leftLabelTransformed, leftLabel, str(saveLeftLabelTransformedPath))
        saveImage(leftImageTransformed, leftImage, str(saveLeftImageTransformedPath))



if __name__=="__main__":
    args = ParseArgs()
    main(args)
