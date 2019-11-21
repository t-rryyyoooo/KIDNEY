import argparse
import os

args = None
def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("textFolder")
    parser.add_argument("suffix")

    args = parser.parse_args()

    return args

def main(args):
    textFolder = os.path.expanduser(args.textFolder)

    with open(textFolder + "/testing_" + args.suffix +  ".txt") as f1:
        with open(textFolder + "/training_" + args.suffix + ".txt") as f2:
            with open(textFolder + "/validation_" + args.suffix + ".txt") as f3:
                file1 = f1.readlines()
                file2 = f2.readlines()
                file3 = f3.readlines()
                x = len(file1)
                y = len(file2)
                z = len(file3)
                print("Suffix : ", args.suffix, x, y, z, x + y + z)

if __name__ == "__main__":
    args = parseArgs()
    main(args)
