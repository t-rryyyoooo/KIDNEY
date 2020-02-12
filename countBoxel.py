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
    args = parser.parse_args()
    return args


def main(args):


    testing = ['019', '023', '054', '093', '096', '123', '127', '136', '141', '153', '188', '191', '201', '001', '017', '020', '022', '043', '082', '094', '115', '120', '137', '173', '174', '205']
    testing =  ['019', '023', '054', '093', '096', '123', '127', '136', '141', '153', '188', '191', '201']
    validation =  ['001', '017', '020', '022', '043', '082', '094', '115', '120', '137', '173', '174', '205']


    print("Patient\tnotAligned\taligned")
    for i, x in enumerate(testing):


        path1 = args.path1 + '/case_00' + x + "/label.mha"
        path2 = args.path2 + "/case_00" + x + '/label.mha' 


        label1 = sitk.ReadImage(path1)
        label2 = sitk.ReadImage(path2)

        label1Array = sitk.GetArrayFromImage(label1)
        label2Array = sitk.GetArrayFromImage(label2)

        count1 = (label1Array == 2).sum()
        count2 = (label2Array == 2).sum()


        print("{}\t{}\t{}".format(i + 1, count1, count2))
        


        

if __name__ == "__main__":
    args = parseArgs()
    main(args)
    
