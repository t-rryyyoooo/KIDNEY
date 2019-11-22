import os
import shutil
import argparse

testing =  ['173', '002', '068', '133', '155', '114', '090', '105', '112', '175', '183', '208', '029', '065', '157', '162', '141', '062', '031', '156', '189', '135', '020', '077', '000', '009', '198', '036']

args = None

def parseArags():
    parser = argparse.ArgumentParser()
    parser.add_argument("originalPath", help="~/Desktop/data/kits19/case_00")
    parser.add_argument("resultPath", help="~/Desktop/data/hist/segmentation/case_00")
    parser.add_argument("savePath", help="~/Desktop/data/save/case_00")
    args = parser.parse_args()

    return args

def main(args):
    originalPath = os.path.expanduser(args.originalPath)
    resultPath = os.path.expanduser(args.resultPath)
    savePath = os.path.expanduser(args.savePath)

    for x in testing:

        ctPath = originalPath + x + "/imaging.nii.gz"
        labelPath = originalPath + x + "/segmentation.nii.gz"
        resultxPath = resultPath + x + "/label.mha"

        savexPath = savePath + x

        if not os.path.exists(savexPath):
            print("Make ", savexPath)
            os.makedirs(savexPath, exist_ok = True)

        shutil.copy(ctPath, savexPath)
        shutil.copy(labelPath, savexPath)
        shutil.copy(resultxPath, savexPath)
        
        print("Successfully extracting images in case_00" + x)

if __name__ == "__main__":
    args = parseArags()
    main(args)

    