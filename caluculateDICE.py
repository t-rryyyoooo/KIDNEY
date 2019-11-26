import numpy as np
import SimpleITK as sitk
import os
import argparse

args = None

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('trueLabel', help = '~/Desktop/data/KIDNEY/case_00')
    parser.add_argument('resultLabel',help = '~/Desktop/data/hist/segmentation/case_00')

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
    testing = ['173', '002', '068', '133', '155', '114', '090', '105', '112', '175', '183', '208', '029', '065', '157', '162', '141', '062', '031', '156', '189', '135', '020', '077', '000', '009', '198', '036']

    wholeDICE=[]
    kidneyDICE = []
    cancerDICE = []
    for x in testing:

        trueLabel = os.path.expanduser(args.trueLabel) + x + '/segmentation.nii.gz'
        resultLabel = os.path.expanduser(args.resultLabel) + x + '/label.mha'

        true = sitk.ReadImage(trueLabel)
        result = sitk.ReadImage(resultLabel)

        trueArray = sitk.GetArrayFromImage(true)
        resultArray = sitk.GetArrayFromImage(result)
     
        wholeDICE.append(DICE(trueArray,resultArray))

        trueKid = np.where(trueArray == 1, 1, 0)
        trueCan = np.where(trueArray == 2, 2, 0)

        resultKid = np.where(resultArray == 1, 1, 0)
        resultCan = np.where(resultArray == 2, 2, 0)


        kidneyDICE.append(DICE(trueKid,resultKid))
        cancerDICE.append(DICE(trueCan,resultCan))
        print('case_00' + x)
        print("Average whole: {}  ".format(DICE(trueArray,resultArray)))
        print("Average kidney: {}  ".format(DICE(trueKid,resultKid)))
        print("Average cancer: {}  ".format(DICE(trueCan,resultCan)))

    print("Average whole: {}  ".format(caluculateAVG(wholeDICE)))
    print("Average kidney: {}  ".format(caluculateAVG(kidneyDICE)))
    print("Average cancer: {}  ".format(caluculateAVG(cancerDICE)))
    print()

if __name__ == '__main__':
    args = parseArgs()
    main(args)