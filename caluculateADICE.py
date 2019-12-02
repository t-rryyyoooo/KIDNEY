import numpy as np
import SimpleITK as sitk
import os
import argparse

args = None

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('trueLabel', help = '~/Desktop/data/KIDNEY/case_00000/segmentation.nii.gz')
    parser.add_argument('resultLabel',help = '~/Desktop/data/hist/segmentation/label.mha')

    args = parser.parse_args()

    return args

def DICE(trueLabel, result):
    intersection=np.sum(np.minimum(np.equal(trueLabel,result),trueLabel))
    union = np.count_nonzero(trueLabel)+np.count_nonzero(result)
    dice = 2 * intersection / union
   
    return dice

def caluculateAVG(num):
    if len(num) == 0:
        return 1.0
    
    else: 
        nsum = 0
        for i in range(len(num)):
            nsum += num[i]

        return nsum / len(num)

def main(args):
    trueLabel = os.path.expanduser(args.trueLabel) 
    resultLabel = os.path.expanduser(args.resultLabel) 

    true = sitk.ReadImage(trueLabel)
    result = sitk.ReadImage(resultLabel)

    trueArray = sitk.GetArrayFromImage(true)
    resultArray = sitk.GetArrayFromImage(result)
    
    trueKid = np.where(trueArray == 1, 1, 0)
    trueCan = np.where(trueArray == 2, 2, 0)

    resultKid = np.where(resultArray == 1, 1, 0)
    resultCan = np.where(resultArray == 2, 2, 0)
   
    print("Average whole: {}  ".format(DICE(trueArray,resultArray)))
    print("Average kidney: {}  ".format(DICE(trueKid,resultKid)))
    print("Average cancer: {}  ".format(DICE(trueCan,resultCan)))


if __name__ == '__main__':
    args = parseArgs()
    main(args)