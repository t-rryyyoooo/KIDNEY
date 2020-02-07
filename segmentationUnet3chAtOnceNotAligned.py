import SimpleITK as sitk
import numpy as np
import argparse
import os
import sys
import tensorflow as tf
from functions import createParentPath, ResampleSize, cancer_dice, kidney_dice, penalty_categorical
from pathlib import Path
from clip3D import searchBound, searchBoundAndMakeIndex, adjustDiff
args = None

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("imageDirectory", help="Labelfile")
    parser.add_argument("modelweightfile", help="Trained model weights file (*.hdf5).")
    parser.add_argument("savePath", help="Segmented label file.(.mha)")
    parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=0, type=int)
    parser.add_argument("-b", "--batchsize", help="Batch size", default=1, type=int)
    parser.add_argument("--expansion", default=0, type=int)

    args = parser.parse_args()
    return args

def main(_):
#    config = tf.compat.v1.ConfigProto()
#    config.gpu_options.allow_growth = True
#    config.allow_soft_placement = True
#    #config.log_device_placement = True
#    sess = tf.compat.v1.Session(config=config)
#    tf.compat.v1.keras.backend.set_session(sess)
#
#    modelweightfile = os.path.expanduser(args.modelweightfile)
#
#    with tf.device('/device:GPU:{}'.format(args.gpuid)):
#        print('loading U-net model {}...'.format(modelweightfile), end='', flush=True)
#        # with open(args.modelfile) as f:
#        #     model = tf.compat.v1.keras.models.model_from_yaml(f.read())
#        # model.load_weights(args.modelweightfile)
#        model = tf.compat.v1.keras.models.load_model(modelweightfile,
#         custom_objects={'penalty_categorical' : penalty_categorical, 'kidney_dice':kidney_dice, 'cancer_dice':cancer_dice})
#
#        print('done')

    savePath = os.path.expanduser(args.savePath)
    createParentPath(savePath)

    meta = {}

    ## Read image
    labelFile = Path(args.imageDirectory) / "segmentation.nii.gz"
    imageFile = Path(args.imageDirectory) / "imaging.nii.gz"

    label = sitk.ReadImage(str(labelFile))
    image = sitk.ReadImage(str(imageFile))

    labelArray = sitk.GetArrayFromImage(label)
    imageArray = sitk.GetArrayFromImage(image)

    startIdx, endIdx = searchBound(labelArray, 'sagittal')
    startIdx, endIdx = adjustDiff(startIdx, endIdx, labelArray.shape[0])

    startIdx -= args.expansion
    endIdx += args.expansion

    clipLabelArray = {"left" : labelArray[startIdx[0] : endIdx[0] + 1,...], 
                      "right" : labelArray[startIdx[1] : endIdx[1] + 1,...]}
    clipImageArray = {"left" : imageArray[startIdx[0] : endIdx[0] + 1,...], 
                      "right" : imageArray[startIdx[1] : endIdx[1] + 1,...]}
    meta["left"] = {"sagittal" : [startIdx[0], endIdx[0] + 1]}
    meta["right"] = {"sagittal" : [startIdx[1], endIdx[1] + 1]}


    print("Clipping images in sagittal direction...")

    startIdx, endIdx = searchBoundAndMakeIndex(clipLabelArray, "axial")
    startIdx, endIdx = adjustDiff(startIdx, endIdx, labelArray.shape[2])

    startIdx -= args.expansion
    endIdx += args.expansion

    clipLabelArray = {"left" : clipLabelArray["left"][..., startIdx[0] : endIdx[0] + 1], 
                      "right" : clipLabelArray["right"][..., startIdx[1] : endIdx[1] + 1]}
    clipImageArray = {"left" : clipImageArray["left"][..., startIdx[0] : endIdx[0] + 1], 
                      "right" : clipImageArray["right"][..., startIdx[1] : endIdx[1] + 1]}

    meta["left"]["axial"] = [startIdx[0], endIdx[0] + 1]
    meta["right"]["axial"] = [startIdx[1], endIdx[1] + 1]


    print("Clipping images in axial direction...")

    startIdx, endIdx = searchBoundAndMakeIndex(clipLabelArray, "coronal")
    startIdx, endIdx = adjustDiff(startIdx, endIdx, labelArray.shape[1])
    startIdx -= args.expansion
    endIdx += args.expansion

    clipLabelArray = {"left" : clipLabelArray["left"][:, startIdx[0] : endIdx[0] + 1, :], 
                      "right" : clipLabelArray["right"][:, startIdx[1] : endIdx[1] + 1, :]}
    clipImageArray = {"left" : clipImageArray["left"][:, startIdx[0] : endIdx[0] + 1, :], 
                      "right" : clipImageArray["right"][:, startIdx[1] : endIdx[1] + 1, :]}

    meta["left"]["coronal"] = [startIdx[0], endIdx[0] + 1] 
    meta["left"]["size"] = clipImageArray["left"].shape
    meta["right"]["coronal"] = [startIdx[1], endIdx[1] + 1]
    meta["right"]["size"] = clipImageArray["right"].shape
    


    print("Clipping images in coronal direction...")
    print("Left Kidney clipped size : {}".format(clipLabelArray["left"].shape))
    print("Right Kidney clipped size : {}".format(clipLabelArray["right"].shape))

                
    clipLabelArray["right"] = clipLabelArray["right"][::-1,...]
    clipImageArray["right"] = clipImageArray["right"][::-1,...]


    clipLabelArray["left"] = sitk.GetImageFromArray(clipLabelArray["left"])
    clipLabelArray["right"] = sitk.GetImageFromArray(clipLabelArray["right"])
    clipImageArray["left"] = sitk.GetImageFromArray(clipImageArray["left"])
    clipImageArray["right"] = sitk.GetImageFromArray(clipImageArray["right"])

    blackArray = np.zeros_like(labelArray)
    for d in ["left", "right"]:
        clipImageArray[d].SetDirection(label.GetDirection())
        clipImageArray[d].SetOrigin(label.GetOrigin())
        clipImageArray[d].SetSpacing(label.GetSpacing())
        clipLabelArray[d].SetDirection(label.GetDirection())
        clipLabelArray[d].SetOrigin(label.GetOrigin())
        clipLabelArray[d].SetSpacing(label.GetSpacing())
        
        length = clipImageArray[d].GetSize()[0]

        clipLabelArray[d] = ResampleSize(clipLabelArray[d], (length, 256, 256))
        clipImageArray[d] = ResampleSize(clipImageArray[d], (length, 256, 256))
        clipImgArray = sitk.GetArrayFromImage(clipImageArray[d])
        clipLabArray = sitk.GetArrayFromImage(clipLabelArray[d])
        zero = np.zeros((256, 256)) - 1024.
        stackedArrayList = []
        for x in range(length):
            if x == 0:
                top = zero
                middle = clipImgArray[..., x]
                bottom = clipImgArray[..., x + 1]

            elif x == (length - 1):
                top = clipImgArray[..., x - 1]
                middle = clipImgArray[..., x]
                bottom = zero

            else:
                top = clipImgArray[..., x - 1]
                middle = clipImgArray[..., x]
                bottom = clipImgArray[..., x + 1]


            stackedArray = np.dstack([top, middle, bottom])
            stackedArray = stackedArray[np.newaxis, ...]

            print('Shape of input image: {}'.format(stackedArray.shape))
            print('segmenting...')
            #paarry = model.predict(stackedArray, batch_size = args.batchsize, verbose = 0)
            print('stackedArray.shape: {}'.format(stackedArray.shape))
            #labelarry = np.argmax(paarry, axis=-1).astype(np.uint8)
            #labelarry = labelarry.reshape(256,256)
            
            labelarry = clipLabArray[..., x]

            stackedArrayList.append(labelarry)

        stackedArray = np.dstack(stackedArrayList)
        
        if d == "right":
            stackedArray = stackedArray[::-1,...]

        stacked = sitk.GetImageFromArray(stackedArray)
        stacked.SetDirection(clipImageArray[d].GetDirection())
        stacked.SetOrigin(clipImageArray[d].GetOrigin())
        stacked.SetSpacing(clipImageArray[d].GetSpacing())

        stacked = ResampleSize(stacked, meta[d]["size"][::-1], is_label=True)
        stackedArray = sitk.GetArrayFromImage(stacked)
        
        print(meta)
        blackArray[meta[d]["sagittal"][0] : meta[d]["sagittal"][1], 
                   meta[d]["coronal"][0] : meta[d]["coronal"][1],
                   meta[d]["axial"][0] : meta[d]["axial"][1]] = stackedArray

    black = sitk.GetImageFromArray(blackArray)
    black.SetDirection(label.GetDirection())
    black.SetOrigin(label.GetOrigin())
    black.SetSpacing(label.GetSpacing())

    sitk.WriteImage(black, savePath, True)
    print("Saving image to {}...".format(savePath))



if __name__ == '__main__':
    args = ParseArgs()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]])
