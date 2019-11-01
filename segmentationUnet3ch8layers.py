import sys
import os
import tensorflow as tf
import numpy as np
import argparse
import SimpleITK as sitk

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("PathList", help="Input image file")#imagepath
    parser.add_argument("modelfile", help="U-net model file (*.yml).")
    parser.add_argument("modelweightfile", help="Trained model weights file (*.hdf5).")
    parser.add_argument("outfile", help="Segmented label file.")
    parser.add_argument("--paoutfile", help="The filename of the estimated probabilistic map file.")
    parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=0, type=int)
    parser.add_argument("-b", "--batchsize", help="Batch size", default=1, type=int)
    args = parser.parse_args()
    return args


def createParentPath(filepath):
    head, _ = os.path.split(filepath)
    if len(head) != 0:
        os.makedirs(head, exist_ok = True)

def ReadSliceDataList3ch_1ch(filename):
    datalist = []
    with open(filename) as f:
        for line in f:
            labelfile, imagefile = line.strip().split('\t')
            datalist.append((imagefile, labelfile))
            
            
    pathDicImg = {}#{~/case_00000/image0 : ~/case_00000/image0_00.mha}
    labellist = {}
    pathList = []#3枚ごとにまとめられたリスト(image,label)

    #パスを同じ腎臓ごとにまとめる
    for path in datalist:
        dicPathI, filePathI = os.path.split(path[0])
        fI,nameI = filePathI.split("_")
        fPathI = os.path.join(dicPathI, fI)
        

        if fPathI not in pathDicImg:
            pathDicImg[fPathI] = []
            labellist[fPathI] = []

        pathDicImg[fPathI].append(path[0])

        labellist[fPathI].append(filePathI)

    #同じ腎臓の中で、あるスライスと前後2枚をくっつける(path)

    for (keyI, valueI),(labkey, labvalue) in zip(pathDicImg.items(), labellist.items()):
        valueI = sorted(valueI)
        labvalue = sorted(labvalue)
        for x in range(len(valueI)):
            if x-1<0 and x+1>=len(valueI):
                v = []
                v.append(None)
                v.append(valueI[x])
                v.append(None)
                pathList.append((v, labvalue[x]))
                
            elif x-1<0:
                vv = []
                vv.append(None)
                vv.append(valueI[x])
                vv.append(valueI[x+1])
                pathList.append((vv, labvalue[x]))
            
            elif x+1>=len(valueI):
                vvv =[]
                vvv.append(valueI[x-1])
                vvv.append(valueI[x])
                vvv.append(None)
                pathList.append((vvv, labvalue[x]))
                
            else:
                pathList.append((valueI[x-1:x+2], labvalue[x]))
         
    return pathList


def ImportImage3ch(pList):#[["case_00000/image0_00.mha", "case_00000/image0_01.mha", "case_00000/image0_02.mha"],\
                          # ["case_00000/image0_01.mha", "case_00000/image0_02.mha", "case_00000/image0_03.mha"]...]
    
    check = False
    nono = False
    
    for x in pList:
        if x is not None:
            img = sitk.ReadImage(x)
            imgArray = sitk.GetArrayFromImage(img)
            
        if x is None and not check:
            nono = True
            continue
        
        if x is not None and not check and nono:
            check = True
            img = sitk.ReadImage(x)
            imgArray = sitk.GetArrayFromImage(img)
            non = np.zeros_like(imgArray)
            stackedArray = non
        
        if x is None and check:
            imgArray = np.zeros_like(stackedArray[:,:,0])
            
        if not check:
            check = True
            stackedArray = imgArray

        else:
            stackedArray = np.dstack([stackedArray, imgArray])
    
    
    return stackedArray, img


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    #config.log_device_placement = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

   # print('loading input image {}...'.format(args.imagefile), end='', flush=True)
    

    

    with tf.device('/device:GPU:{}'.format(args.gpuid)):
        print('loading U-net model {}...'.format(args.modelfile), end='', flush=True)
        with open(args.modelfile) as f:
            model = tf.keras.models.model_from_yaml(f.read())
        model.load_weights(args.modelweightfile)
        print('done')

    createParentPath(args.outfile)

    dataList = ReadSliceDataList3ch_1ch(args.PathList)
    for data in dataList:
        imagearry, inputimage = ImportImage3ch(data[0])
        
        print('Shape of input image: {}'.format(imagearry.shape))
        imagearry = imagearry[np.newaxis,...]
        print('Shape of input image: {}'.format(imagearry.shape))
        print('segmenting...')
        paarry = model.predict(imagearry, batch_size = args.batchsize, verbose = 1)
        print('paarry.shape: {}'.format(paarry.shape))
        labelarry = np.argmax(paarry, axis=-1).astype(np.uint8)
        
        labelarry = labelarry.reshape(256,256)
        print('labelarry.shape: {}'.format(labelarry.shape))
        
        print('saving segmented label to {}...'.format(os.path.join(args.outfile,data[1])), end='', flush=True)
        segmentation = sitk.GetImageFromArray(labelarry)
        segmentation.SetOrigin(inputimage.GetOrigin())
        segmentation.SetSpacing(inputimage.GetSpacing())
        segmentation.SetDirection(inputimage.GetDirection())
        
        sitk.WriteImage(segmentation, os.path.join(args.outfile, data[1]), True)
        print('done')

        if args.paoutfile is not None:
            createParentPath(args.paoutfile)
            print('saving PA to {}...'.format(args.paoutfile), end='', flush=True)
            pa = sitk.GetImageFromArray(paarry)
            pa.SetOrigin(inputimage.GetOrigin())
            pa.SetSpacing(inputimage.GetSpacing())
            pa.SetDirection(inputimage.GetDirection())
            
            sitk.WriteImage(pa, args.paoutfile)
            print('done')


if __name__ == '__main__':
    args = ParseArgs()
    tf.app.run(main=main, argv=[sys.argv[0]])
