import argparse
import SimpleITK as sitk
from pathlib import Path
import numpy as np
from functions import createParentPath
args = None

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("labelPath", help="~/Desktop/data/slice/summed_hist_1.0/path")
    parser.add_argument("savePath", help="~/Desktop/data/textList")
    args = parser.parse_args()
    return args


def main(args):
    testing =  ['019', '023', '054', '093', '096', '123', '127', '136', '141', '153', '188', '191', '201']

    for x in testing:
        truePath = Path(args.labelPath) / ("case_00" + x) / "label.nii.gz"
        segPath = Path(args.labelPath) / ("case_00" + x) /"label_seg.mha"

        true = sitk.ReadImage(str(truePath))
        seg = sitk.ReadImage(str(segPath))

        trueArray = sitk.GetArrayFromImage(true)
        segArray = sitk.GetArrayFromImage(seg)

        check = np.logical_and(trueArray == segArray, segArray == 2)

        overlapArray = np.where(check, 3, segArray)
        overlap = sitk.GetImageFromArray(overlapArray)
        overlap.SetDirection(true.GetDirection())
        overlap.SetOrigin(true.GetOrigin())
        overlap.SetSpacing(true.GetSpacing())

        savePath = Path(args.savePath) / ("case_00" + x) / "label_over.mha"

        createParentPath(savePath)

        sitk.WriteImage(overlap, str(savePath), True)




        

if __name__ == "__main__":
    args = parseArgs()
    main(args)
    
