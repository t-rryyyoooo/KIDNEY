import argparse
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import sys
import os

validation = ['134', '046', '021', '038', '044', '070', '179', '006', '204', '152', '190', '084', '118', '047', '200', '101', '148', '050', '110', '032', '078', '025', '016', '142', '168', '111', '182', '041']
training = ['207', '080', '095', '099', '007', '028', '053', '137', '176', '106', '083', '127', '094', '103', '193', '019', '197', '037', '196', '035', '113', '146', '185', '093', '145', '102', '056', '042', '139', '067', '180', '061', '026', '174', '153', '001', '064', '075', '091', '129', '147', '058', '178', '085', '086', '203', '003', '138', '144', '051', '122', '024', '076', '205', '121', '063', '108', '027', '188', '184', '004', '160', '119', '164', '045', '130', '072', '049', '166', '154', '209', '143', '013', '163', '074', '081', '048', '052', '126', '087', '149', '117', '136', '012', '206', '040', '191', '054', '124', '066', '195', '187', '132', '057', '150', '060', '089', '104', '170', '159', '171', '169', '039', '125', '199', '011', '008', '073', '055', '107', '079', '092', '192', '030', '186', '181', '088', '172', '034', '018', '120', '082', '177', '014', '158', '109', '100', '131', '033', '010', '140', '069', '022', '123', '071', '023', '098', '116', '128', '043', '059', '161', '115', '097', '167', '017', '015', '201', '096', '202']
testing = ['173', '002', '068', '133', '155', '114', '090', '105', '112', '175', '183', '208', '029', '065', '157', '162', '141', '062', '031', '156', '189', '135', '020', '077', '000', '009', '198', '036']
ignore = ['005','151','165','194']

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("slicePath", help='/Volumes/Untitled/slice/hist_1.0')
    parser.add_argument("npyFile", help='histogram(.npy)')
    parser.add_argument("savePath")
    parser.add_argument("alpha",type=float)
    parser.add_argument("-n", '--number', help='The number of patients', default=210, type=int)

    

    args = parser.parse_args()
    return args

def createParentPath(filepath):
    head, _ = os.path.split(filepath)
    if len(head) != 0:
        os.makedirs(head, exist_ok = True)

def write_file(file_name, text):
    if not os.path.exists(file_name):
        createParentPath(file_name)
    with open(file_name, mode='a') as file:
        #print(text)
        file.write(text + "\n")

def main(args):
    
    print('Loading npy file...')
    npyFile = os.path.expanduser(args.npyFile)
    HIST = np.load(npyFile)
    print('Loading it has done. ')
    aHIST = HIST * args.alpha + (1 - args.alpha) / 2048

    #Make CDF
    cdf = aHIST.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    temp = (cdf_m - cdf_m.min())/(cdf_m.max()-cdf_m.min())
    cdf_m = 2048*temp
    cdf = np.ma.filled(cdf_m,0).astype('int64')
    print('Making CDF has done.' )

    #Read iamge and equalizing histogram
    for x in range(args.number):
        sx = str(x).zfill(3)
        if sx in ignore:
            continue
        
        #Make CT and segmentation path
        slicePath = os.path.expanduser(args.slicePath)
        pI = Path(slicePath) / ('image/case_00' + sx)
        pL = Path(slicePath) / ('label/case_00' + sx)
        
        if not pI.exists():
            print('File does not exist. ')
            sys.exit()
        
        if not pL.exists():
            print('File does not exist. ')
            sys.exit()
        
        #pI = list(pI.iterdir()).sort()

        for i,l in zip(sorted(pI.iterdir()), sorted(pL.iterdir())):
            #savePath
            savePath = os.path.expanduser(args.savePath)
            OPI = Path(savePath) / "image" / ('case_00' + sx) / i.name
            OPL = Path(savePath) / "label" / ('case_00' + sx) / l.name
            OPT = Path(savePath) / "path" / ("case_00" + sx + ".txt")

            #Make parent path
            if not OPI.parent.exists():
                createParentPath(str(OPI))
            
            if not OPL.parent.exists():
                createParentPath(str(OPL))

            if not OPT.parent.exists():
                createParentPath(str(OPT))
                                                                
            #print(str(OPL) + "\t" + str(OPI))
            
            ctImg = sitk.ReadImage(str(i))
            ctImgArray = sitk.GetArrayFromImage(ctImg)
            ctImgArray = ctImgArray + 1024
            ctImgArray = np.clip(ctImgArray, 0, 2048)
            
            #Equalizing histogram
            ctImgArray = cdf[ctImgArray] - 1024
            
            #Save ct image
            ctImgNew = sitk.GetImageFromArray(ctImgArray)
            ctImgNew.SetOrigin(ctImg.GetOrigin())
            ctImgNew.SetSpacing(ctImg.GetSpacing())
            ctImgNew.SetDirection(ctImg.GetDirection())
            
            sitk.WriteImage(ctImgNew, str(OPI))
            
            #Save label image
            labImg = sitk.ReadImage(str(l))
            sitk.WriteImage(labImg, str(OPL), True)
            
            #Save textfile
            
            write_file(str(OPT), str(OPL) + "\t" + str(OPI))

        print("case_00" + sx + "done. ")


if __name__ == '__main__':
    args = ParseArgs()
    main(args)