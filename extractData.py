import os
import shutil
import argparse

args = None
def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("ctPath")
    parser.add_argument("labelPath")
    parser.add_argument("segmentationPath")
    parser.add_argument("savePath")

    args = parser.parse_args()
    return args

def createParentPath(filepath):
    head, _ = os.path.split(filepath)
    if len(head) != 0:
        os.makedirs(head, exist_ok = True)

def main(args):
    if not os.path.exists(args.savePath):
        print("Make ", args.savePath)
        os.makedirs(args.savePath, exist_ok = True)
        
    shutil.copy(args.ctPath, args.savePath)
    shutil.copy(args.labelPath, args.savePath)
    shutil.copy(args.segmentationPath, args.savePath)

if __name__=="__main__":
    args = ParseArgs()
    main(args)