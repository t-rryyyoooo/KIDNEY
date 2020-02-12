import sys
import os
import argparse
from functions import readlines_file, save_file, list_file
import numpy as np
import SimpleITK as sitk
args = None

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("path1", help="~/Desktop/data/slice/summed_hist_1.0/path")
    parser.add_argument("path2")
    parser.add_argument("path3")
    args = parser.parse_args()
    return args


def main(args):


    testing = ['019', '023', '054', '093', '096', '123', '127', '136', '141', '153', '188', '191', '201', '001', '017', '020', '022', '043', '082', '094', '115', '120', '137', '173', '174', '205']
    #testing =  ['019', '023', '054', '093', '096', '123', '127', '136', '141', '153', '188', '191', '201']
    validation =  ['001', '017', '020', '022', '043', '082', '094', '115', '120', '137', '173', '174', '205']


    print("Patient\tnotAligned\taligned\ttrue")
    s1 = 0
    s2 = 0
    for i, x in enumerate(testing):


        path1 = args.path1 + '/case_00' + x + "/label.mha"
        path2 = args.path2 + "/case_00" + x + '/label.mha' 
        path3 = args.path3 + "/case_00" + x + "/segmentation.nii.gz"


        label1 = sitk.ReadImage(path1)
        label2 = sitk.ReadImage(path2)
        label3 = sitk.ReadImage(path3)

        label1Array = sitk.GetArrayFromImage(label1)
        label2Array = sitk.GetArrayFromImage(label2)
        label3Array = sitk.GetArrayFromImage(label3)

        count1 = (label1Array == 2).sum()
        count2 = (label2Array == 2).sum()
        count3 = (label3Array == 2).sum()

        s1 += count1
        s2 += count2


        print("{}\t{}\t{}\t{}".format(i + 1, count1, count2, count3))

    print("Mean")
    print(s1 / len(testing), s2 / len(testing))
        


        

if __name__ == "__main__":
    args = parseArgs()
    main(args)
    
