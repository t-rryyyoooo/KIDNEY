import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import copy
import cv2
import os
import sys

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("labelfile", help="Labelfile")
    parser.add_argument("imagefile", help="imagefile")
    parser.add_argument("-l", "--layers", help="Number of laywers", default=5, type=int)
    parser.add_argument("savelabelpath", help="savelabelpath")
    parser.add_argument("saveimagepath", help="saveimagepath")
    parser.add_argument("textfilepath", help="save textfile path")
    parser.add_argument("--nobn", help="Do not use batch normalization layer", action='store_true')
    parser.add_argument("--nodropout", help="Do not use dropout layer", action='store_true')
    parser.add_argument("--noaugmentation", help="Do not use training data augmentation", action='store_true')
    parser.add_argument("--magnification", help="Magnification coefficient for data augmentation", default=10, type=int)
    parser.add_argument("--latestfile", help="The filename of the latest weights.")
    parser.add_argument("--bestfile", help="The filename of the best weights.")
    parser.add_argument("--weightinterval", help="The interval between checkpoint for weight saving.", type=int)
    parser.add_argument("--weightfile", help="The filename of the trained weight parameters file for fine tuning or resuming.")
    parser.add_argument("--premodel", help="The filename of the previously trained model")
    parser.add_argument("--initialepoch", help="Epoch at which to start training for resuming a previous training", default=0, type=int)
    #parser.add_argument("--idlist", help="The filename of ID list for splitting input datasets into training and validation datasets.")
    #parser.add_argument("--split", help="Fraction of the training data to be used as validation data.", default=0.0, type=float)
    parser.add_argument("--logdir", help="Log directory", default='log')
    parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=0, type=int)
    parser.add_argument("--history")

    args = parser.parse_args()
    return args


def write_file(file_name, text):
    with open(file_name, 'a') as file:
        file.write(text + "\n")

def cut3D(labelArray, imageArray, axis):
    if axis=="axial":
        totalNumber = len(labelArray[0,0,:])
        imageIndex = []

    #高さ方向の腎臓、腎臓がんの範囲特定
        for x in range(totalNumber):
            if np.where(labelArray[:,:,x]!=0,True,False).any():
                imageIndex.append(x)

        sliceNumber = int(len(imageIndex)/10)

        #上下数枚を取り込み
        IndexFirst = imageIndex[0]
        IndexFinal = imageIndex[-1]
        imageIndexFirst = []
        imageIndexFinal = []

        for a in range(sliceNumber):
            if (IndexFirst-a-1)>=0:
                imageIndexFirst.append(IndexFirst-a-1)
            if (IndexFinal+a+1)<totalNumber:
                imageIndexFinal.append(IndexFinal+a+1)

        imageIndexFirst.reverse()
        imageIndex= imageIndexFirst + imageIndex + imageIndexFinal#高さ方向の必要な部分のインデックスの配列

        #高さ方向について必要な部分のみ切り取った3D画像を生成
        cut3DLabelArray = labelArray[:,:,imageIndex[0]:imageIndex[-1]+1]
        cut3DImageArray = imageArray[:,:,imageIndex[0]:imageIndex[-1]+1]

        return cut3DLabelArray, cut3DImageArray, imageIndex, sliceNumber
    
    if axis=="coronal":
        totalNumber = len(labelArray[0,:,:0])
        imageIndex = []

    #高さ方向の腎臓、腎臓がんの範囲特定
        for x in range(totalNumber):
            if np.where(labelArray[:,x,:]!=0,True,False).any():
                imageIndex.append(x)

        sliceNumber = 15

        #上下数枚を取り込み
        IndexFirst = imageIndex[0]
        IndexFinal = imageIndex[-1]
        imageIndexFirst = []
        imageIndexFinal = []

        for a in range(sliceNumber):
            if (IndexFirst-a-1)>=0:
                imageIndexFirst.append(IndexFirst-a-1)
            if (IndexFinal+a+1)<totalNumber:
                imageIndexFinal.append(IndexFinal+a+1)

        imageIndexFirst.reverse()
        imageIndex= imageIndexFirst + imageIndex + imageIndexFinal#高さ方向の必要な部分のインデックスの配列

        #高さ方向について必要な部分のみ切り取った3D画像を生成
        cut3DLabelArray = labelArray[:,imageIndex[0]:imageIndex[-1]+1,:]
        cut3DImageArray = imageArray[:,imageIndex[0]:imageIndex[-1]+1,:]
        
        return cut3DLabelArray, cut3DImageArray,imageIndex
        
    if axis=="sagittal":
        totalNumber = len(imageArray[:,0,0])
        imageIndex = []

    #高さ方向の腎臓、腎臓がんの範囲特定
        for x in range(totalNumber):
            if np.where(labelArray[x,:,:]!=0,True,False).any():
                imageIndex.append(x)

        sliceNumber = 15

        #上下数枚を取り込み
        IndexFirst = imageIndex[0]
        IndexFinal = imageIndex[-1]
        imageIndexFirst = []
        imageIndexFinal = []

        for a in range(sliceNumber):
            if (IndexFirst-a-1)>=0:
                imageIndexFirst.append(IndexFirst-a-1)
            if (IndexFinal+a+1)<totalNumber:
                imageIndexFinal.append(IndexFinal+a+1)

        imageIndexFirst.reverse()
        imageIndex= imageIndexFirst + imageIndex + imageIndexFinal#高さ方向の必要な部分のインデックスの配列

        #高さ方向について必要な部分のみ切り取った3D画像を生成
        cut3DLabelArray = labelArray[imageIndex[0]:imageIndex[-1]+1,:,:]
        cut3DImageArray = imageArray[imageIndex[0]:imageIndex[-1]+1,:,:]
        
        return cut3DLabelArray, cut3DImageArray,imageIndex

def cut_image(imgArray, paddingSize=15, center=None, wh=None, angle=None):#paddingSize(int), center(tuple), wh=(tuple)
    #unsigned int に変換
    area = 0
    if center==None and wh==None and angle==None :
        imgArray = np.array(imgArray,dtype=np.uint8)

        ## 輪郭抽出
        contours, hierarchy = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        # 面積が最大の輪郭を選択する。
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        area = cv2.contourArea(cnt)
        
        # 外接矩形を取得する。

        center, wh, angle = cv2.minAreaRect(cnt)
    
    if center==None and wh!=None and angle==None :
        imgArray = np.array(imgArray,dtype=np.uint8)

        ## 輪郭抽出
        contours, hierarchy = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        # 面積が最大の輪郭を選択する。
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        area = cv2.contourArea(cnt)
        
        # 外接矩形を取得する。

        center, _, angle = cv2.minAreaRect(cnt)
    
    

    #intに変換
    center = list(center)
    wh = list(wh)
    center = tuple(map(lambda x: int(x), center))
    wh = tuple(map(lambda x: int(x), wh))#wh = (width, height)
    paddedwh =  tuple(map(lambda x: int(x) + paddingSize, wh))#wh = (width, height)

    

    imgwh = (imgArray.shape[1], imgArray.shape[0])#元画像の幅と高さ

    #スケールを指定
    scale = 1.0

    #画像を回転させるための行列を生成
    trans = cv2.getRotationMatrix2D(center, angle , scale)

    #アフィン変換（元画像の行列にさっきの行列をかけて画像を回転）
    rotatedImg = cv2.warpAffine(imgArray, trans, imgwh)
    
    # 切り出す。
    x0 = center[1] - int(paddedwh[1]/2) 
    x1 = center[1] + int(paddedwh[1]/2) 
    y0 = center[0] - int(paddedwh[0]/2)
    y1 = center[0] + int(paddedwh[0]/2)
    
    if x0<0:
        x0 = 0
    if y0<0:
        y0 = 0
    roi = rotatedImg[x0 : x1 , y0 : y1]
    
    if area==0:
        return roi, center, wh, angle
    
    else:
        ##情報出力
        #print("Area: ",area)
        #print("Center: ",center)
        #print("Original width and height: ",wh)
        #print("Angle: ",angle)
        #print("Padding_size: ",paddingSize)
        #print("Padded width and height: ", paddedwh)
        #print("Original image shape: ",imgArray.shape)
        #print("Rotated image shape: ",rotatedImg.shape)
        #print("ROI image shape: ", roi.shape)
        #print("\n")
    
        return roi, center, wh, angle
    
def caluculate_area(imgArray):
    area = 0
    #unsigned int に変換
    imgArray = np.array(imgArray,dtype=np.uint8)

    ## 輪郭抽出
    contours, hierarchy = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0
    
    
    # 面積が最大の輪郭を選択する。
    cnt = max(contours, key=lambda x: cv2.contourArea(x))
    area = cv2.contourArea(cnt)
    
    return area

def Resampling(image, newsize, roisize, origin = None, is_label = False):
    #isize = image.GetSize()
    ivs = image.GetSpacing()
    
    if image.GetNumberOfComponentsPerPixel() == 1:
        minmax = sitk.MinimumMaximumImageFilter()
        minmax.Execute(image)
        minval = minmax.GetMinimum()
    else:
        minval = None
    
    osize = (newsize, newsize )
    

    
    ovs = [ vs * s / os for vs, s, os in zip(ivs, roisize, osize) ]
    

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(osize)
    if origin is not None:
        resampler.SetOutputOrigin(origin)
    else:
        resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputSpacing(ovs)
    if minval is not None:
        resampler.SetDefaultPixelValue(minval)
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    resampled = resampler.Execute(image)

    return resampled

def save_image(imageArray, image, savePath):
    LF = sitk.GetImageFromArray(imageArray)
    LF.SetOrigin(image.GetOrigin())
    LF.SetSpacing(image.GetSpacing())
    sitk.WriteImage(LF, savePath, True)

def main(args):
    ## Read image
    label = sitk.ReadImage(args.labelfile)
    image = sitk.ReadImage(args.imagefile)

    labelArray = sitk.GetArrayFromImage(label)
    imageArray = sitk.GetArrayFromImage(image)

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
        write_file("exceptPatient.txt", args.savelabelpath.rsplit(os.sep)[-1])
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
        #print("Separated size: ",cutKidFragLabel[i].shape)
        
        #axial方向について、３D画像として切り取る
        cutKidFragLabel[i], cutKidFragImage[i], cutIndex, snum = cut3D(cutKidFragLabel[i],cutKidFragImage[i],"axial")
        
        print("cutted size_"+str(i)+": ",cutKidFragLabel[i].shape)

        
        #save_image(cutKidFragLabel[i][:,:,x], label,  "./slice/label_"+str(i)+"_"+str(x).zfill(2)+".mha")
        #save_image(cutKidFragImage[i][:,:,x], image,  "./slice/image_"+str(i)+"_"+str(x).zfill(2)+".mha")
        
        ##最大サイズの腎臓を持つスライスの特定
        mArea = []
        for ckfl in range(len(cutKidFragLabel[i][0,0,:])):
            mArea.append(caluculate_area(cutKidFragLabel[i][:,:,ckfl]))
            maxArea = np.argmax(mArea)
        #print("max index: ", maxArea)
        #print("\n")
        
        #最大サイズのスライスの幅、高さの計算
        #print("Max size data")
        roi, maxCenter, maxwh, maxAngle = cut_image(cutKidFragLabel[i][:,:,maxArea])
        roi, center, wh, angle = cut_image(cutKidFragImage[i][:,:,maxArea], center=maxCenter, wh=maxwh, angle=maxAngle)
        
        rmaxwh = list(maxwh)
        rmaxwh = rmaxwh[::-1]
        rmaxwh = tuple(rmaxwh)#調整
        #print("maxwh: ",maxwh)
        #print("rmaxwh: ",rmaxwh)
        #print("Each data")

        snumber = -1
        layers = args.layers#層の数 
        noKidImg = 0
        check = False

        for ckfl in range(len(cutKidFragLabel[i][0,0,:])):
            a = caluculate_area(cutKidFragLabel[i][:,:,ckfl])
            #print(a)
            #print(maxCenter, maxwh, maxAngle)
            ##腎臓のない領域の画像保存
            if a==0:
                #print(ckfl,"(st,nd,rd,th) slice")
                x0 = maxCenter[1] - int((maxwh[1]+15)/2)
                x1 = maxCenter[1] + int((maxwh[1]+15)/2)
                y0 = maxCenter[0] - int((maxwh[0]+15)/2)
                y1 = maxCenter[0] + int((maxwh[0]+15)/2)
                #print(x0,x1,y0,y1)
                if x0<0:
                    x0 = 0
                if y0<0:
                    y0 = 0
                #print(x0,x1,y0,y1)

                roi_label = cutKidFragLabel[i][x0 :x1, y0 :y1, ckfl]
                roi_image = cutKidFragImage[i][x0 :x1, y0 :y1, ckfl]

                roi_label = cv2.resize(roi_label, (256, 256), 0, 0, cv2.INTER_NEAREST)
                roi_image = cv2.resize(roi_image, (256, 256), 0, 0, cv2.INTER_LINEAR)


                if not np.where(roi_label==2,True, False).any():
                    continue

                if not check:
                    OPL = os.path.join(args.savelabelpath,str(0))
                    OPL = os.path.join(OPL,"label{}_{:02d}.mha".format(i,ckfl))
                    OPI = os.path.join(args.saveimagepath,str(0))
                    OPI = os.path.join(OPI,"image{}_{:02d}.mha".format(i,ckfl))
                    
                    save_image(roi_label, label, OPL)
                    save_image(roi_image, image, OPI)
                    path = os.path.join(args.textfilepath,str(0)+".txt")
                    write_file(path, OPL + "\t" + OPI)
                    
                    noKidImg += 1
                    
                    
                else:
                    
                    OPL = os.path.join(args.savelabelpath,str(layers-1))
                    OPL = os.path.join(OPL,"label{}_{:02d}.mha".format(i,ckfl))
                    OPI = os.path.join(args.saveimagepath,str(layers-1))
                    OPI = os.path.join(OPI,"image{}_{:02d}.mha".format(i,ckfl))
                    
                    save_image(roi_label, label, OPL)
                    save_image(roi_image, image, OPI)
                    path = os.path.join(args.textfilepath,str(layers-1)+".txt")
                    write_file(path,OPL + "\t" + OPI)



                #print("Original shape: ", cutKidFragLabel[i][:,:,ckfl].shape)
                #print("ROI_shape: ", roi_label[ckfl].shape)
                #print("\n")
                
            ##腎臓領域ありの時
            else:
                check = True
                snum = int((len(cutKidFragLabel[i][0,0,:])-2*noKidImg)/(layers-1))
                #print(ckfl,"(st,nd,rd,th) slice")
                roi, center, wh, angle = cut_image(cutKidFragLabel[i][:,:,ckfl])
                if (maxwh[0]>maxwh[1])==(wh[0]>wh[1]):
                    roi, center, wh, angle = cut_image(cutKidFragLabel[i][:,:,ckfl],wh=maxwh)##中心,角度取得
                    roi_label = roi
                    
                else:
                    roi, center, wh, angle = cut_image(cutKidFragLabel[i][:,:,ckfl],wh=rmaxwh)##中心,角度取得
                    roi_label = roi
                    
                    
                roi, center, wh, angle = cut_image(cutKidFragImage[i][:,:,ckfl], center=center, wh=wh, angle=angle)
                roi_image = roi

                roi_label = cv2.resize(roi_label, (256, 256), 0, 0, cv2.INTER_NEAREST)
                roi_image = cv2.resize(roi_image, (256, 256), 0, 0, cv2.INTER_LINEAR)

                
                if not np.where(roi_label==2, True, False).any():
                    continue
                
                
                OPL = os.path.join(args.savelabelpath,"label{}_{:02d}.mha".format(i,ckfl))
                OPI = os.path.join(args.saveimagepath,"image{}_{:02d}.mha".format(i,ckfl))
                
                print(roi_label.shape)
                #print(label.GetSpacing())
                save_image(roi_label, label, OPL)
                save_image(roi_image, image, OPI)
                path = args.textfilepath
                write_file(path, OPL + "\t" + OPI)
        
        #print("The number of images without kidney: ", noKidImg)            
        #print("The number of images with kidney per layer: ",snum) 
          
    ##分けられているかどうかの確認
    if check1 == check2:
        print("Succeeded in cutting.")
        print(args.savelabelpath.rsplit(os.sep)[-1]+" done.\n")

    else:
        print("Failed to cutting.")


if __name__ == '__main__':
    args = ParseArgs()
    main(args)