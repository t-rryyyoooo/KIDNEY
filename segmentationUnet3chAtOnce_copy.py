import SimpleITK as sitk
import numpy as np
import argparse
import copy
import os
import sys
import copy
import tensorflow as tf
from functions import createParentPath, Resampling, cancer_dice, kidney_dice, penalty_categorical
from cut import *
from effect import *
args = None

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("labelfile", help="Labelfile")
    parser.add_argument("imagefile", help="imagefile")
    parser.add_argument("-l", "--layers", help="Number of laywers", default=5, type=int)
    #parser.add_argument("modelfile", help="U-net model file (*.yml).")
    parser.add_argument("modelweightfile", help="Trained model weights file (*.hdf5).")
    parser.add_argument("savepath", help="Segmented label file.(.mha)")
    parser.add_argument("alpha", default=0.0, type=float)
    parser.add_argument("--paoutfile", help="The filename of the estimated probabilistic map file.")
    parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=0, type=int)
    parser.add_argument("-b", "--batchsize", help="Batch size", default=1, type=int)

    
    args = parser.parse_args()
    return args

def main(_):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    #config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    modelweightfile = os.path.expanduser(args.modelweightfile)

    with tf.device('/device:GPU:{}'.format(args.gpuid)):
        print('loading U-net model {}...'.format(modelweightfile), end='', flush=True)
        # with open(args.modelfile) as f:
        #     model = tf.compat.v1.keras.models.model_from_yaml(f.read())
        # model.load_weights(args.modelweightfile)
        model = tf.compat.v1.keras.models.load_model(modelweightfile,
         custom_objects={'penalty_categorical' : penalty_categorical, 'kidney_dice':kidney_dice, 'cancer_dice':cancer_dice})

        print('done')

    savepath = os.path.expanduser(args.savepath)
    createParentPath(savepath)

    labelfile = os.path.expanduser(args.labelfile)
    imagefile = os.path.expanduser(args.imagefile)
    ## Read image
    label = sitk.ReadImage(labelfile)
    image = sitk.ReadImage(imagefile)

    labelArray = sitk.GetArrayFromImage(label)
    imageArray = sitk.GetArrayFromImage(image)

    print("Whole size: ",labelArray.shape)

    totalNumber = len(labelArray[:,0,:0])
    #######スライス開始##########


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
        sys.exit()



    cutKidFragLabel = []#[[1つ目の腎臓の行列],[2つ目の腎臓の行列],..]
    cutKidFragImage = []
    skip_cnt = 0
    check1=0
    check2=0

    inversedImage = np.zeros_like(labelArray)
    invDic = [[] for x in range(2)]##invereに必要な情報を保存
    cutIndex = []##高さ方向の挿入位置

    for i,kidFrag in enumerate(kidIndex):
    
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
        cutKidFragLabel[i], cutKidFragImage[i], cIndex, snum = cut3D(cutKidFragLabel[i],cutKidFragImage[i],"axial")
        
        #print("cutted size_"+str(i)+": ",cutKidFragLabel[i].shape)

        
        
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


                roi_lab = cutKidFragLabel[i][x0 :x1, y0 :y1, ckfl]
                roi_img = cutKidFragImage[i][x0 :x1, y0 :y1, ckfl]
               
                
                center = maxCenter
                wh = maxwh
                angle = 0
                
            ##腎臓領域ありの時
            else:
                roi, center, wh, angle = cut_image(cutKidFragLabel[i][:,:,ckfl])

                if (maxwh[0]>maxwh[1])==(wh[0]>wh[1]):
                    roi, center, wh, angle = cut_image(cutKidFragLabel[i][:,:,ckfl],wh=maxwh)##中心,角度取得
                    roi_lab = roi
                    
                else:
                    roi, center, wh, angle = cut_image(cutKidFragLabel[i][:,:,ckfl],wh=rmaxwh)##中心,角度取得
                    roi_lab = roi
                    
                    
                roi, center, wh, angle = cut_image(cutKidFragImage[i][:,:,ckfl], center=center, wh=wh, angle=angle)
                roi_img = roi
            

            roi_img = np.array(roi_img, dtype=np.int64)

            if roi_img.max()>ctMax:
                ctMax = roi_img.max()
            
            if roi_img.min()<ctMin:
                ctMin = roi_img.min()
            
            imgArrayList.append(roi_img)

            ##ヒストグラム平坦化    
            # roi_img = np.array(roi_img, dtype=np.uint8)
            # roi_img = cv2.equalizeHist(roi_img)

            ##inverse用
            invDic[i].append({
                "roi_lab" : np.zeros_like(roi_lab),
                "roi_img" : roi_img, 
                "cutKidFragLabel" : cutKidFragLabel[i][:,:,ckfl], 
                "wh" : wh, "center" : center, "angle" : angle, 
                "labelArray" : labelArray,
                "image" : label
                })
            print("{}(st nd) kidney {}/{} cutted".format(i,len(invDic[i]),len(cutKidFragLabel[i][0,0,:])))
        
        #ヒストグラム均一化
        equalizedImageArrayList = equalizingHistogram(imgArrayList, args.alpha)
        
        llll = 0
        for equalizedImage, inv in zip(equalizedImageArrayList, invDic[i]):
            inv["roi_img"] = equalizedImage
            # save_image_256(equalizedImage, label, r"test/test"+str(i)+"_"+str(llll)+".mha")
            # llll += 1
        
        cutIndex.append(cIndex)
    
    ####スライス終了####
    
    #####segmentation#########
    for i in range(len(invDic)):
        for l in range(1,len(invDic[i])-1):
            #save_image_256(invDic[i][l]["roi_lab"], label,os.path.join(args.savepath, "label{}_{:02d}.mha".format(i,l)), is_lab=False)
            for m in range(-1,2):
                
                imgArray = invDic[i][l+m]["roi_img"]
                if imgArray.shape!=(256,256):
                    img = sitk.GetImageFromArray(imgArray)
                    img = Resampling(img, (256, 256), img.GetSize())
                    imgArray = sitk.GetArrayFromImage(img)
                  
                if m==-1:
                    stackedArray = imgArray

                else:
                    stackedArray = np.dstack([stackedArray, imgArray])
           
            stackedArray = stackedArray[np.newaxis,...]
            print('Shape of input image: {}'.format(stackedArray.shape))
            print('segmenting...')
            paarry = model.predict(stackedArray, batch_size = args.batchsize, verbose = 0)
            print('paarry.shape: {}'.format(paarry.shape))
            labelarry = np.argmax(paarry, axis=-1).astype(np.uint8)
            labelarry = labelarry.reshape(256,256)

            #save_image_256(labelarry, label,os.path.join(args.savepath, "result{}_{:02d}.mha".format(i,l)), is_lab=True)
            
            lab = sitk.GetImageFromArray(labelarry)
            lab = Resampling(lab, invDic[i][l]["roi_img"].shape[::-1], lab.GetSize())
            labelarry = sitk.GetArrayFromImage(lab)
            print('labelarry.shape: {}'.format(labelarry.shape))
            invDic[i][l]["roi_lab"] = labelarry
            
            ##invDic[i][l]["roi_lab"]<-segmented image
    
    ###########################Inverse#############################################
    
    icheck = False
    for i in range(len(invDic)):
        scheck = False
        for l in range(len(invDic[i])):
            #save_image_256(invDic[i][l]["roi_lab"], label,os.path.join(args.savepath, "result{}_{:02d}.mha".format(i,l)), is_lab=True)

            iImg = inverse_image(invDic[i][l]["roi_lab"], 
                invDic[i][l]["cutKidFragLabel"], 
                invDic[i][l]["wh"], invDic[i][l]["center"], invDic[i][l]["angle"], 
                invDic[i][l]["labelArray"], i)
            
            ##高さ方向にstack
            if not scheck:
                scheck = True
                inversedImg = iImg
            
            else:
                inversedImg = np.dstack([inversedImg, iImg])


        print("inversed",inversedImg.shape)
        
        if i==0:
            icheck = True
            
            iImage = copy.copy(inversedImage)
        
            inversedImage[:,:,cutIndex[i][0]:cutIndex[i][-1]+1] = inversedImg
            
        
        else:
            iImage[:,:,cutIndex[i][0]:cutIndex[i][-1]+1] = inversedImg
            
            inversedImage = inversedImage+iImage
        
        
        
            
    LF = sitk.GetImageFromArray(inversedImage)
    
    LF.SetOrigin(label.GetOrigin())
    LF.SetSpacing(label.GetSpacing())
    LF.SetDirection(label.GetDirection())
    
    print('saving segmented label to {}...'.format(savepath), end='', flush=True)
    
    sitk.WriteImage(LF, savepath, True)
        
        #print("The number of images without kidney: ", noKidImg)            
        #print("The number of images with kidney per layer: ",snum)

    

    


if __name__ == '__main__':
    args = ParseArgs()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]])
