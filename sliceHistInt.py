import SimpleITK as sitk
import numpy as np
import argparse
import copy
import os
import sys
from functions import createParentPath, write_file, Resampling
from cut import *
from effect import *

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("originalFilePath", help="$HOME/Desktop/data/kits19/case_00000")
    parser.add_argument("savePath", help="$HOME/Desktop/data/slice/hist_0.0")

    parser.add_argument("alpha", default=0.0, type=float)

    args = parser.parse_args()
    return args


def main(args):
    print(args.alpha)

    labelFile = Path(args.originalFilePath) / 'segmentation.nii.gz'
    imageFile = Path(args.originalFilePath) / 'imaging.nii.gz'

    ## Read image
    label = sitk.ReadImage(str(labelFile))
    image = sitk.ReadImage(str(imageFile))

    labelArray = sitk.GetArrayFromImage(label)
    imageArray = sitk.GetArrayFromImage(image)

    meta = {}
    meta["Spacing"] = label.GetSpacing()[1:3]
    meta["Origin"] = label.GetOrigin()[1:3]
    meta["Direction"] = label.GetDirection()[3:5] + label.GetDirection()[6:8]


    print("Whole size: ",labelArray.shape)

    totalNumber = len(labelArray[:,0,:0])

    #################腎臓の数を特定, １ブロックにつき１つになるように3Dのまま、最大範囲切り取る##########################


    kidIndex = []#特定された腎臓のインデックス群[[1つ目],[2つ目],...]
    kidFragment = []#分けた3腎臓のインデックス群の保存先
    k = []#一時的保持


    for x in range(totalNumber-1):
        if np.where(labelArray[x,:,:]!=0,True,False).any() and np.where(labelArray[x+1,:,:]!=0,True,False).any():
            k.append(x)

        elif np.where(labelArray[x,:,:]!=0,True,False).any() and not(np.where(labelArray[x+1,:,:]!=0,True,False).any()):
            k.append(x)
            kidIndex.append(copy.copy(k))
            k.clear()

    if len(kidIndex) != 2:
        print("The patient has horse shoe kidney.")
        print(args.savelabelpath.rsplit(os.sep)[-1]+" failed.")
        #write_file("exceptPatient.txt", args.savelabelpath.rsplit(os.sep)[-1])
        sys.exit()



    cutKidFragLabel = []#[[1つ目の腎臓の行列],[2つ目の腎臓の行列],..]
    cutKidFragImage = []
    skip_cnt = 0
    check1=0
    check2=0

    for i,kidFrag in enumerate(kidIndex):
        roi_label = []
        roi_image = []
        IndexFirst = kidFrag[0]
        IndexFinal = kidFrag[-1]

        

        if IndexFirst < len(labelArray[:,0,0])/2:
            kidFragment.append(np.arange(IndexFinal+1,len(labelArray[:,0,0])))
            check1 += 1
        else:
            kidFragment.append(np.arange(IndexFirst))
            check2 += 1

        
        
        #分けた腎臓の行列を保存
        if kidFragment[i][0]==kidFragment[i][-1]:
            skip_cnt -= 1
            continue

        else:
            cutKidFragLabel.append(labelArray[kidFragment[i][0]:kidFragment[i][-1]+1,:,:])
            cutKidFragImage.append(imageArray[kidFragment[i][0]:kidFragment[i][-1]+1,:,:])

        i = i + skip_cnt

        #一つの腎臓を反転
        if i == 1:
            cutKidFragLabel[1] = cutKidFragLabel[1][::-1,:,:]
            cutKidFragImage[1] = cutKidFragImage[1][::-1,:,:]



        #############################################################################################
        
        #axial方向について、３D画像として切り取る
        cutKidFragLabel[i], cutKidFragImage[i], _, _ = cut3D(cutKidFragLabel[i],cutKidFragImage[i],"axial")

        print("cutted size_"+str(i)+": ",cutKidFragLabel[i].shape)

        ##最大サイズの腎臓を持つスライスの特定
        mArea = []
        for ckfl in range(len(cutKidFragLabel[i][0,0,:])):
            mArea.append(caluculate_area(cutKidFragLabel[i][:,:,ckfl]))
            maxArea = np.argmax(mArea)
        
        #最大サイズのスライスの幅、高さの計算
        roi, maxCenter, maxwh, maxAngle = cut_image(cutKidFragLabel[i][:,:,maxArea])
        roi, center, wh, angle = cut_image(cutKidFragImage[i][:,:,maxArea], center=maxCenter, wh=maxwh, angle=maxAngle)
        
        rmaxwh = list(maxwh)
        rmaxwh = rmaxwh[::-1]
        rmaxwh = tuple(rmaxwh)#調整

        imgArrayList = []
        ctMax = -10**9
        ctMin = 10**9
        for ckfl in range(len(cutKidFragLabel[i][0,0,:])):
           

            a = caluculate_area(cutKidFragLabel[i][:,:,ckfl])
            
            ##腎臓のない領域の画像保存
            if a==0:
                x0 = maxCenter[1] - int((maxwh[1]+15)/2)
                x1 = maxCenter[1] + int((maxwh[1]+15)/2)
                y0 = maxCenter[0] - int((maxwh[0]+15)/2)
                y1 = maxCenter[0] + int((maxwh[0]+15)/2)
                if x0<0:
                    x0 = 0
                if y0<0:
                    y0 = 0

                roi_label = cutKidFragLabel[i][x0 :x1, y0 :y1, ckfl]
                roi_image = cutKidFragImage[i][x0 :x1, y0 :y1, ckfl]

            ##腎臓領域ありの時
            else:
                roi, center, wh, angle = cut_image(cutKidFragLabel[i][:,:,ckfl])
                if (maxwh[0]>maxwh[1])==(wh[0]>wh[1]):
                    roi, center, wh, angle = cut_image(cutKidFragLabel[i][:,:,ckfl],wh=maxwh)##中心,角度取得
                    roi_label = roi
                    
                else:
                    roi, center, wh, angle = cut_image(cutKidFragLabel[i][:,:,ckfl],wh=rmaxwh)##中心,角度取得
                    roi_label = roi
                    
                    
                roi, center, wh, angle = cut_image(cutKidFragImage[i][:,:,ckfl], center=center, wh=wh, angle=angle)
                roi_image = roi
                
            
            #ヒストグラム平坦化
            roi_image = np.array(roi_image, dtype=np.int64)
            if roi_image.max()>ctMax:
                ctMax = roi_image.max()
            
            if roi_image.min()<ctMin:
                ctMin = roi_image.min()
            
            imgArrayList.append(roi_image)

    


            ##1枚1枚ヒストグラム平坦化    
            #roi_image = np.array(roi_image, dtype=np.uint8)
            #roi_image = cv2.equalizeHist(roi_image)

            patientID = args.originalFilePath.split('/')[-1]
            OPL = Path(args.savePath) / 'label' / patientID / "label{}_{:02d}.mha".format(i,ckfl)
            OPI = Path(args.savePath) / 'image' / patientID / "image{}_{:02d}.mha".format(i,ckfl)
            OPT = Path(args.savePath) / 'path' / (patientID + '.txt')

            #Make parent path
            if not OPI.parent.exists():
                createParentPath(str(OPI))
            
            if not OPL.parent.exists():
                createParentPath(str(OPL))

            if not OPT.parent.exists():
                createParentPath(str(OPT))


            save_image_256(roi_label, label, str(OPL), is_lab=True)
            #save_image_256(roi_image, image, OPI)
            
            write_file(str(OPT), str(OPL) + "\t" + str(OPI))
        
        
        print(ctMax, ctMin)    
        
        equalizedImageArrayList = equalizingHistogram(imgArrayList, args.alpha)
        for x ,equalizedImageArray in enumerate(equalizedImageArrayList):
            OPI = Path(args.savePath) / 'image' / patientID / "image{}_{:02d}.mha".format(i,x)
            
            save_image_256(equalizedImageArray, image , str(OPI))

    ##分けられているかどうかの確認
    if check1 == check2:
        print("Succeeded in cutting.")
        print(args.originalFilePath.split('/')[-1]+" done.\n")

    else:
        print("Failed to cutting.")


if __name__ == '__main__':
    args = ParseArgs()
    main(args)