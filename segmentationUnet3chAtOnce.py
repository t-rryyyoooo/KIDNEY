import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import copy
import cv2
import os
import sys
import copy
import tensorflow as tf

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

def cancer_dice(y_true, y_pred):
    K = tf.keras.backend

    eps = K.constant(1e-6)
    truelabels = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    
    truelabel = K.cast(K.equal(truelabels, 2), tf.int32)##ガンだけ
    prediction = K.cast(K.equal(predictions, 2), tf.int32)

    intersection = K.cast(K.sum(K.minimum(K.cast(K.equal(prediction, truelabel), tf.int32), truelabel)), tf.float32)
    union = tf.count_nonzero(prediction, dtype=tf.float32) + tf.count_nonzero(truelabel, dtype=tf.float32)
    dice = 2 * intersection / (union + eps)
    return dice

def kidney_dice(y_true, y_pred):#canver
    K = tf.keras.backend

    eps = K.constant(1e-6)
    truelabels = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    
    truelabel = K.cast(K.equal(truelabels, 1), tf.int32)##腎臓だけ
    prediction = K.cast(K.equal(predictions, 1), tf.int32)

    intersection = K.cast(K.sum(K.minimum(K.cast(K.equal(prediction, truelabel), tf.int32), truelabel)), tf.float32)
    union = tf.count_nonzero(prediction, dtype=tf.float32) + tf.count_nonzero(truelabel, dtype=tf.float32)
    dice = 2 * intersection / (union + eps)
    return dice

def penalty_categorical(y_true,y_pred):
    K = tf.keras.backend
    
    array_tf = tf.convert_to_tensor(y_true,dtype=tf.float32)
    pred_tf = tf.convert_to_tensor(y_pred,dtype=tf.float32)

    epsilon = K.epsilon()

    result = tf.reduce_sum(array_tf,[0,1,2,3])

    #result_pow = tf.pow(result,1.0/3.0)
    result_pow = tf.math.log(result)

    weight_y = result_pow / tf.reduce_sum(result_pow)

    k_dice = kidney_dice(y_true, y_pred)
    c_dice = cancer_dice(y_true, y_pred)

    return (-1) * tf.reduce_sum( 1 / (weight_y + epsilon) * array_tf * tf.log(pred_tf + epsilon),axis=-1) \
       + (1 - k_dice) + (1 - c_dice)

def createParentPath(filepath):
    head, _ = os.path.split(filepath)
    if len(head) != 0:
        os.makedirs(head, exist_ok = True)

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
        vcheck = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(vcheck)==3:
            _, contours, _ = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        else:
            contours, _ = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        # 面積が最大の輪郭を選択する。
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        area = cv2.contourArea(cnt)
        
        # 外接矩形を取得する。

        center, wh, angle = cv2.minAreaRect(cnt)
    
    if center==None and wh!=None and angle==None :
        imgArray = np.array(imgArray,dtype=np.uint8)

        ## 輪郭抽出
        vcheck = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(vcheck)==3:
            _, contours, _ = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        else:
            contours, _ = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
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
   
    vvcheck = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(vvcheck)==3:
        _, contours, _ = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    

    else:
        contours, _ = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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
    
    osize = newsize
    

    
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

def makeCDF(imgArrayList, alpha = 0):#imgArrayList:int64
    imgArrayList = [ x + 1024 for x in imgArrayList ]
    
    imgArrayList = [np.where(x<0, 0, x) for x in imgArrayList]
    imgArrayList = [np.where(x>2048, 2048,x) for x in imgArrayList]
    
    ctRange = 2048 + 1
    #uniformDistribution = np.array([ctRange]*ctRange)
    
    HIST = np.array([0.0]*ctRange)
    for x in imgArrayList:
        hist, _  = np.histogram(x.flatten(), ctRange, [0, 2048+1])
        HIST += hist
    
    
    print(HIST)
    HIST /= sum(HIST)
    HIST = HIST * (1 - alpha) + alpha / 2048
    print(HIST)
    cdf = HIST.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    temp = (cdf_m - cdf_m.min())/(cdf_m.max()-cdf_m.min())
    cdf_m = 2048*temp
    cdf = np.ma.filled(cdf_m,0).astype('int64')
    
    return cdf,imgArrayList

def equalizingHistogram(imgArrayList, alpha):
    cdf, imgArrayList = makeCDF(imgArrayList, alpha)
    
    equalizedImageArrayList = []
    for imgAr in imgArrayList:
        x = cdf[imgAr] - 1024
        equalizedImageArrayList.append(x)
    
    return equalizedImageArrayList

def save_image_256(imageArray, image, savePath, is_lab=False):
    LF = sitk.GetImageFromArray(imageArray)
    if is_lab:
        LF = Resampling(LF,(256,256),LF.GetSize(), is_label=True)
    else:
        LF = Resampling(LF,(256,256),LF.GetSize())

    LF.SetOrigin(image.GetOrigin())
   # LF.SetSpacing(image.GetSpacing())
    LF.SetSpacing(LF.GetSpacing())
    sitk.WriteImage(LF, savePath, True)

def inverse_image(roi, cutKidFragLabel, wh, center, angle,labelArray, i):
    blackImg = np.zeros_like(cutKidFragLabel)

    wh = tuple([x+y for x,y in zip(wh,(15, 15))])

    x0 = center[1] - int(wh[1]/2) 
    x1 = center[1] + int(wh[1]/2) 
    y0 = center[0] - int(wh[0]/2)
    y1 = center[0] + int(wh[0]/2)

    if x0<0:
        x0 = 0
    if y0<0:
        y0 = 0


    blackImg[x0 : x1 , y0 : y1] = roi
    imgwh = (blackImg.shape[1], blackImg.shape[0])#元画像の幅と高さ


    #画像を回転させるための行列を生成
    trans = cv2.getRotationMatrix2D(center, (-1)*angle , 1.0)

    #アフィン変換（元画像の行列にさっきの行列をかけて画像を回転）
    iImg = cv2.warpAffine(blackImg, trans, imgwh)

    margin = np.zeros((abs(imgwh[1]-labelArray.shape[1]), imgwh[0]))

    
    if center[0]<imgwh[1]/2:
        iImg = np.vstack([iImg, margin])

    else:
        iImg = np.vstack([margin, iImg])

    if i==1:
        iImg = iImg[::-1,:]
    
    return iImg

def main(_):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    #config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    with tf.device('/device:GPU:{}'.format(args.gpuid)):
        print('loading U-net model {}...'.format(args.modelfile), end='', flush=True)
        # with open(args.modelfile) as f:
        #     model = tf.compat.v1.keras.models.model_from_yaml(f.read())
        # model.load_weights(args.modelweightfile)
        model = tf.compat.v1.keras.models.load_model(args.modelweightfile,
         custom_objects={'penalty_categorical' : penalty_categorical, 'kidney_dice':kidney_dice, 'cancer_dice':cancer_dice})

        print('done')

    createParentPath(args.savepath)


    ## Read image
    label = sitk.ReadImage(args.labelfile)
    image = sitk.ReadImage(args.imagefile)

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
        print(args.savelabelpath.rsplit(os.sep)[-1]+" failed.")
        write_file("exceptPatient.txt", args.savelabelpath.rsplit(os.sep)[-1])
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
    
    print('saving segmented label to {}...'.format(args.savepath), end='', flush=True)
    sitk.WriteImage(LF, args.savepath, True)
        
        #print("The number of images without kidney: ", noKidImg)            
        #print("The number of images with kidney per layer: ",snum)

    

    


if __name__ == '__main__':
    args = ParseArgs()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]])