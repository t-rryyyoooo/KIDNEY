import numpy as np
import SimpleITK as sitk
import os
import argparse
from functions import DICE, caluculateAVG
from pathlib import Path
args = None

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help = '~/Desktop/data/box')
    parser.add_argument('prefix',help = '3ch')

    args = parser.parse_args()

    return args


def main(args):
    testing =  ['019', '023', '054', '093', '096', '123', '127', '136', '141', '153', '188', '191', '201']
    
    #path ='/home/vmlab/Desktop/data/box'
    path = args.path
    prefix = args.prefix

    path = Path(path)
    segmentation = sorted(path.glob('**/segmentation_*.mha'))

    for s in segmentation:
        newPath = s.parent / (prefix + "_" + str(s.name))
        print("Rename {} to {}...".format(str(s), str(newPath)))
    
        s.replace(newPath)
        
        print("Done.")

if __name__ == '__main__':
    args = parseArgs()
    main(args)
