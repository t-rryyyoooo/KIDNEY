import SimpleITK as sitk
import numpy as np
import tensorflow.python.keras.backend as K
import os
import copy
import argparse
import sys

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("labelfile", help="The filename of input label image.")
    parser.add_argument("imagefile", help="The filename(s) of input image image(s).")
    parser.add_argument("outputlabelfile", help="The filename of output label image.")
    parser.add_argument("outputimagefile", help="The filename(s) of output image image(s).")
    parser.add_argument("pathlistfile", help="The filename(s) of pathlist.")
    parser.add_argument("-o", "--listfile", help="The filename of the list of extracted slice and organ existence list.", default='listfile.txt')
    
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

        sliceNumber = int(len(imageIndex)/7)#特定されたスライスの枚数/7

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

        sliceNumber = int(len(imageIndex)/7)#特定されたスライスの枚数/7

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

        sliceNumber = int(len(imageIndex)/7)#特定されたスライスの枚数/7

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

def centerOfGravity(mat):
    hei, wid = mat.shape
    tilex = np.arange(wid,dtype=float)
    tiley = np.arange(hei,dtype=float)
    tx = np.tile(tilex,[hei,1])
    ty = np.tile(tiley.T,[wid,1]).T
    Sum = np.sum(mat)
    ax = np.sum(mat*tx)/Sum
    ay = np.sum(mat*ty)/Sum
    return [int(ay),int(ax)]

def cut2D(labelArray, imageArray, axis):#入力：2D画像のlabel, imageと軸
    if axis=="coronal":
        totalNumber = len(labelArray[0,:])
        imageIndex = []

    #coronal方向の腎臓、腎臓がんの範囲特定
        for x in range(totalNumber):
            if np.where(labelArray[:,x]!=0,True,False).any():
                imageIndex.append(x)

        #sliceNumber = int(len(imageIndex)/7)#特定されたスライスの枚数/7
        sliceNumber = 10
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
        cut2DLabelArray = labelArray[:,imageIndex[0]:imageIndex[-1]+1]
        cut2DImageArray = imageArray[:,imageIndex[0]:imageIndex[-1]+1]
        
        return cut2DLabelArray, cut2DImageArray,imageIndex
        
    if axis=="sagittal":
        totalNumber = len(imageArray[:,0])
        imageIndex = []

    #高さ方向の腎臓、腎臓がんの範囲特定
        for x in range(totalNumber):
            if np.where(labelArray[x,:]!=0,True,False).any():
                imageIndex.append(x)

        #sliceNumber = int(len(imageIndex)/7)#特定されたスライスの枚数/7
        sliceNumber = 10
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
        cut2DLabelArray = labelArray[imageIndex[0]:imageIndex[-1]+1,:]
        cut2DImageArray = imageArray[imageIndex[0]:imageIndex[-1]+1,:]
        
        return cut2DLabelArray, cut2DImageArray,imageIndex

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

def main(args):
    pathlist = []#保存先のパスのリスト

    ## Read image
    label = sitk.ReadImage(args.labelfile)
    image = sitk.ReadImage(args.imagefile)
    
    labelArray = sitk.GetArrayFromImage(label)
    imageArray = sitk.GetArrayFromImage(image)
    
    

    totalNumber = len(labelArray[:,0,:0])
    cut3DLabelArray = []#切り取った3D画像の行列の保存先(label)
    cut3DImageArray = []#切り取った3D画像の行列の保存先(image)
    
    
    #################腎臓の数を特定, １ブロックにつき１つになるように3Dのまま、最大範囲切り取る##########################


    kidIndex = []#特定された腎臓のインデックス群[[1つ目],[2つ目],...]
    kidFragment = []#分けた3腎臓のインデックス群の保存先
    k = []#一時的保持
    

    for x in range(totalNumber):
        if np.where(labelArray[x,:,:]!=0,True,False).any() and np.where(labelArray[x+1,:,:]!=0,True,False).any():
            k.append(x)
        
        elif np.where(labelArray[x,:,:]!=0,True,False).any() and not(np.where(labelArray[x+1,:,:]!=0,True,False).any()):
            k.append(x)
            kidIndex.append(copy.copy(k))
            k.clear()
        
    if len(kidIndex) != 2:
        write_file("exceptPatient.txt", args.outputlabelfile.rsplit(os.sep)[-1])
        sys.exit()
        


    cutKidFragLabel = []#[[1つ目の腎臓の行列],[2つ目の腎臓の行列],..]
    cutKidFragImage = []
    skip_cnt = 0
    
    for i,kidFrag in enumerate(kidIndex):
        IndexFirst = kidFrag[0]
        IndexFinal = kidFrag[-1]
        
        #print(kidFrag)
        
        if i==0:
            kidFragment.append(np.arange(IndexFinal+1,len(labelArray[:,0,0])))
        else:
            kidFragment.append(np.arange(IndexFirst))
    
        
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
            
            
    #############################################################################################################
        
        #axial方向について、３D画像として切り取る
        cutKidFragLabel[i], cutKidFragImage[i], cutIndex, snum = cut3D(cutKidFragLabel[i],cutKidFragImage[i],"axial")
    
        snumber = 0 #スライス保存先を決めるための変数
        
        for l in range(len(cutIndex)):
            if l < snum:
                
                ######2D保存######## 
                LF = sitk.GetImageFromArray(cutKidFragLabel[i][:,:,l])
                LF = Resampling(LF,256,LF.GetSize())
                IF = sitk.GetImageFromArray(cutKidFragImage[i][:,:,l])
                IF = Resampling(IF,256,IF.GetSize())
                
                OPL = os.path.join(args.outputlabelfile,str(0))
                OPL = os.path.join(OPL,"label{}_{}.mha".format(i,l))
                OPI = os.path.join(args.outputimagefile,str(0))
                OPI = os.path.join(OPI,"image{}_{}.mha".format(i,l))
                
                path = os.path.join(args.pathlistfile,str(0)+".txt")
                write_file(path,OPL + "\t" + OPI)

                LF.SetOrigin(label.GetOrigin())
                IF.SetOrigin(image.GetOrigin())
                LF.SetSpacing(label.GetSpacing())
                IF.SetSpacing(image.GetSpacing())
                sitk.WriteImage(LF, OPL, True)
                sitk.WriteImage(IF, OPI, True)
            
            elif l < len(cutIndex)-snum:
                if (l-snum)%snum==0 and snumber<8:
                    snumber += 1

                hei, wid = cutKidFragLabel[i][:,:,l].shape
                center = [int(hei/2), int(wid/2)]#画像の中心
                gravity = centerOfGravity(np.where(cutKidFragLabel[i][:,:,l]!=0,1,0))#重心
                d = [c-g for c, g in zip(center,gravity)]#重心と画像の中心の差
                
                ##腎臓の重心が画像の中心に来るように移動
                slideLabel = np.roll(cutKidFragLabel[i][:,:,l],tuple(d),axis=(0,1))
                slideImage = np.roll(cutKidFragImage[i][:,:,l],tuple(d),axis=(0,1))
                
                slideLabel, slideImage,_ = cut2D(slideLabel, slideImage, axis="sagittal")
                slideLabel, slideImage,_ = cut2D(slideLabel, slideImage, axis="coronal")
                
                
                ######2D保存#######  
                LF = sitk.GetImageFromArray(slideLabel)
                LF = Resampling(LF,256,LF.GetSize())
                IF = sitk.GetImageFromArray(slideImage)
                IF = Resampling(IF,256,IF.GetSize())
                
                

                OPL = os.path.join(args.outputlabelfile,str(snumber))
                OPL = os.path.join(OPL,"label{}_{}.mha".format(i,l))
                OPI = os.path.join(args.outputimagefile,str(snumber))
                OPI = os.path.join(OPI,"image{}_{}.mha".format(i,l))
                
                path = os.path.join(args.pathlistfile,str(snumber)+".txt")
                write_file(path,OPL + "\t" + OPI)

                LF.SetOrigin(label.GetOrigin())
                IF.SetOrigin(image.GetOrigin())
                LF.SetSpacing(label.GetSpacing())
                IF.SetSpacing(image.GetSpacing())
                sitk.WriteImage(LF, OPL, True)
                sitk.WriteImage(IF, OPI, True)
                
            else:
                
                ######2D保存########    
                LF = sitk.GetImageFromArray(cutKidFragLabel[i][:,:,l])
                LF = Resampling(LF,256,LF.GetSize())
                IF = sitk.GetImageFromArray(cutKidFragImage[i][:,:,l])
                IF = Resampling(IF,256,IF.GetSize())

                OPL = os.path.join(args.outputlabelfile,str(9))
                OPL = os.path.join(OPL,"label{}_{}.mha".format(i,l))
                OPI = os.path.join(args.outputimagefile,str(9))
                OPI = os.path.join(OPI,"image{}_{}.mha".format(i,l))
                
                path = os.path.join(args.pathlistfile,str(9)+".txt")
                write_file(path,OPL + "\t" + OPI)

                LF.SetOrigin(label.GetOrigin())
                IF.SetOrigin(image.GetOrigin())
                LF.SetSpacing(label.GetSpacing())
                IF.SetSpacing(image.GetSpacing())
                sitk.WriteImage(LF, OPL, True)
                sitk.WriteImage(IF, OPI, True)
            
    for r in pathlist:
        write_file(args.pathlistfile,r)

       
    print("number of slices: ",len(cutIndex))
    print("batch size: ",snum)
    print(args.outputlabelfile.rsplit(os.sep)[-1]," done.")
    

if __name__=="__main__":
    args = ParseArgs()
    main(args)