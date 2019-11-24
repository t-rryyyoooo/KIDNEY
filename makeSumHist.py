import SimpleITK as sitk
import numpy as np
from pathlib import Path
import sys
import argparse

training = ['207', '080', '095', '099', '007', '028', '053', '137', '176', '106', '083', '127', '094', '103', '193', '019', '197', '037', '196', '035', '113', '146', '185', '093', '145', '102', '056', '042', '139', '067', '180', '061', '026', '174', '153', '001', '064', '075', '091', '129', '147', '058', '178', '085', '086', '203', '003', '138', '144', '051', '122', '024', '076', '205', '121', '063', '108', '027', '188', '184', '004', '160', '119', '164', '045', '130', '072', '049', '166', '154', '209', '143', '013', '163', '074', '081', '048', '052', '126', '087', '149', '117', '136', '012', '206', '040', '191', '054', '124', '066', '195', '187', '132', '057', '150', '060', '089', '104', '170', '159', '171', '169', '039', '125', '199', '011', '008', '073', '055', '107', '079', '092', '192', '030', '186', '181', '088', '172', '034', '018', '120', '082', '177', '014', '158', '109', '100', '131', '033', '010', '140', '069', '022', '123', '071', '023', '098', '116', '128', '043', '059', '161', '115', '097', '167', '017', '015', '201', '096', '202']

validation = ['134', '046', '021', '038', '044', '070', '179', '006', '204', '152', '190', '084', '118', '047', '200', '101', '148', '050', '110', '032', '078', '025', '016', '142', '168', '111', '182', '041']
testing = ['173', '002', '068', '133', '155', '114', '090', '105', '112', '175', '183', '208', '029', '065', '157', '162', '141', '062', '031', '156', '189', '135', '020', '077', '000', '009', '198', '036']
ignore = ['005','151','165','194']


args = None
def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('directoryPath', help='/home/vmlab/Desktop/data/slice/hist_1.0/image/case_00')
    parser.add_argument('savePath')

    args = parser.parse_args()

    return args

def main(args):
    ctRange = 2048 + 1
    HIST = np.array([0.0] * ctRange)

    for x in range(210):
        sx = str(x).zfill(3)
        if sx not in training:
            continue
            
        p = Path('/home/vmlab/Desktop/data/slice/hist_1.0/image/case_00' + sx)
        if not p.exists():
            print('File does not exist. ')
            sys.exit()
        for i in p.glob('image*'):
            img = sitk.ReadImage(str(i))
            imgArray = sitk.GetArrayFromImage(img)
            imgArray = imgArray + 1024
            imgArray = np.clip(imgArray, 0, 2048)
            hist, _ = np.histogram(imgArray.ravel(), ctRange, [0, ctRange], density=True)
            HIST += hist
        print('case_00' + sx + 'done. ')
        
    HIST = HIST / HIST.sum()

    #HIST = HIST * args.alpha + (1 - args.alpha) / 2048


    if np.isnan(HIST).any():
        print('Failed to make histogram')
    else:
        print('Suceeded in making histogram, now saving histogram...')
        np.save(args.savePath, HIST)
        print('Successfully saved to ' + args.savePath)


    



if __name__ == '__main__':
    args = ParseArgs()
    main(args)