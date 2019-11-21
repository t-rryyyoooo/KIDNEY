import os
import shutil
import argparse

testing =  ['173', '002', '068', '133', '155', '114', '090', '105', '112', '175', '183', '208', '029', '065', '157', '162', '141', '062', '031', '156', '189', '135', '020', '077', '000', '009', '198', '036']

args = None

def parseArags():
    parser = argparse.ArgumentParser()
    parser.add_argument("originalPath")
    parser.add_argument("resultPath")
    parser.add_argument("savePath")
    args = parser.parse_args

    return args

def main(args):
    for x in testing:
        originalPath = os.path.expanduser(args.originalPath)
        resultPath = os.path.expanduser(args.resultPath)
        savePath = os.path.expanduser(args.savePath)

        ctPath = originalPath + x + "/imaging.nii.gz"
        labelPath = originalPath + x + "/segmentation.nii.gz"
        resultPath = resultPath + x + "/label.mha"

        savePath = savePath + x

        if not os.path.exists(savePath):
            print("Make ", savePath)
            os.makedirs(savePath, exist_ok = True)

        shutil.copy(ctPath, savePath)
        shutil.copy(labelPath, savePath)
        shutil.copy(resultPath, savePath)
        
        print("Successfully extracting images in case_00" + x)

if __name__ == "__main__":
    args = parseArags()
    main(args)

    