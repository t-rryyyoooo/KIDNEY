import numpy as np
import SimpleITK as sitk
import os
import argparse
from functions import DICEVersion2, caluculateAVG

args = None

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('trueLabel', help = '~/Desktop/data/KIDNEY')
    parser.add_argument('resultLabel',help = '~/Desktop/data/hist/segmentation')

    args = parser.parse_args()

    return args


def main(args):
#    testing =  ['019', '023', '054', '093', '096', '123', '127', '136', '141', '153', '188', '191', '201']
    testing = ['173', '002', '068', '133', '155', '114', '090', '105', '112', '175', '183', '208', '029', '065', '157', '162', '141', '062', '031', '156', '189', '135', '020', '077', '000', '009', '198', '036']

    wholeDICE=[]
    kidneyDICE = []
    cancerDICE = []
    for x in testing:

        trueLabelLeft = os.path.expanduser(args.trueLabel) + '/case_00' + x + '/label_left.nii.gz'
        resultLabelLeft = os.path.expanduser(args.resultLabel) + '/case_00' + x + '/segmentation_left.mha'
        trueLabelRight = os.path.expanduser(args.trueLabel) + '/case_00' + x + '/label_right.nii.gz'
        resultLabelRight = os.path.expanduser(args.resultLabel) + '/case_00' + x + '/segmentation_right.mha'


        trueLeft = sitk.ReadImage(trueLabelLeft)
        resultLeft = sitk.ReadImage(resultLabelLeft)
        trueRight = sitk.ReadImage(trueLabelRight)
        resultRight = sitk.ReadImage(resultLabelRight)


        trueArrayLeft = sitk.GetArrayFromImage(trueLeft)
        resultArrayLeft = sitk.GetArrayFromImage(resultLeft)
        trueArrayRight = sitk.GetArrayFromImage(trueRight)
        resultArrayRight = sitk.GetArrayFromImage(resultRight)

        whole = DICEVersion2(trueArrayLeft, resultArrayLeft, trueArrayRight, resultArrayRight)
        wholeDICE.append(whole)

        trueKidLeft = np.where(trueArrayLeft == 1, 1, 0)
        trueCanLeft = np.where(trueArrayLeft == 2, 2, 0)
        trueKidRight = np.where(trueArrayRight == 1, 1, 0)
        trueCanRight= np.where(trueArrayRight == 2, 2, 0)


        resultKidLeft = np.where(resultArrayLeft == 1, 1, 0)
        resultCanLeft = np.where(resultArrayLeft == 2, 2, 0)
        resultKidRight= np.where(resultArrayRight == 1, 1, 0)
        resultCanRight= np.where(resultArrayRight == 2, 2, 0)

       # if (trueCanLeft != 2).all() and (resultCanLeft != 2).all():
       #     cancer = 

            


        kidney = DICEVersion2(trueKidLeft, resultKidLeft, trueKidRight, resultKidRight)
        cancer = DICEVersion2(trueCanLeft, resultCanLeft, trueCanRight, resultCanRight)
        kidneyDICE.append(kidney)
        cancerDICE.append(cancer)

        print('case_00' + x)
        print("Average whole: {}  ".format(whole))
        print("Average kidney: {}  ".format(kidney))
        print("Average cancer: {}  ".format(cancer))

    print("Average whole: {}  ".format(caluculateAVG(wholeDICE)))
    print("Average kidney: {}  ".format(caluculateAVG(kidneyDICE)))
    print("Average cancer: {}  ".format(caluculateAVG(cancerDICE)))
    print()

if __name__ == '__main__':
    args = parseArgs()
    main(args)
