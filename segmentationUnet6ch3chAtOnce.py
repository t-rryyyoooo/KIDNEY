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
from cut import saveImage

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile", help="imagefile")
    #parser.add_argument("modelfile", help="U-net model file (*.yml).")
    parser.add_argument("modelweightfile", help="Trained model weights file (*.hdf5).")
    parser.add_argument("savepath", help="Segmented label file.(.mha)")
    parser.add_argument("alpha", default=0.0, type=float)
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

    imagefile = os.path.expanduser(args.imagefile)
    ## Read image
    image = sitk.ReadImage(imagefile)

    imageArray = sitk.GetArrayFromImage(image)

    print("Whole size: ",imageArray.shape)

    originalShape = imageArray.shape

    resizedImageArray, axis = sliceImage(imageArray, interpolation="linear")
    
    segmentedArray = []
    if axis == 0:
        length = originalShape[0]

        zero = np.zeros((256, 256)) - 1024.

        for x in range(length):
            if x == 0:
                3chArray = np.dstack([zero, resizedImageArray[x, :, :], resizedImageArray[x + 1, :, :]])

            elif x == (length - 1):
                3chArray = np.dstack([resizedImageArray[x - 1, :, :], resizedImageArray[x, :, :], zero])

            else:
                3chArray = np.dstack([resizedImageArray[x - 1, :, :], resizedImageArray[x, :, :], resizedImageArray[x + 1, :, :]])

            
            3chArray = 3chArray[np.newaxis,...]
            print('Shape of input shape: {}'.format(3chArray.shape))
            
            print('segmenting...')
            segArray = model.predict(3chArray, batch_size=args.batchsize, verbose=0)
            segArray = np.argmax(segArray, axis=-1).astype(np.uint8)
            segArray = segArray.reshape(256, 256)

            segmentedArray.append(segArray)

        segmentedArray = np.stack(segmentedArray, axis=axis)
        print('SegmentedArray shape : {}'.format(segmentedArray.shape))

        segmented = sitk.GetImageFromArray(segmentedArray)
        saveImage(segmented, img, savePath)

    
        print('saving segmented label to {}...'.format(savepath), end='', flush=True)
    
    if axis == 1:
        length = originalShape[1]

        zero = np.zeros((256, 256)) - 1024.

        for x in range(length):
            if x == 0:
                3chArray = np.dstack([zero, resizedImageArray[:, x, :], resizedImageArray[:, x + 1, :]])

            elif x == (length - 1):
                3chArray = np.dstack([resizedImageArray[:, x - 1, :], resizedImageArray[:, x, :], zero])

            else:
                3chArray = np.dstack([resizedImageArray[:, x - 1, :], resizedImageArray[:, x, :], resizedImageArray[:, x + 1, :]])

            
            3chArray = 3chArray[np.newaxis,...]
            print('Shape of input shape: {}'.format(3chArray.shape))
            
            print('segmenting...')
            segArray = model.predict(3chArray, batch_size=args.batchsize, verbose=0)
            segArray = np.argmax(segArray, axis=-1).astype(np.uint8)
            segArray = segArray.reshape(256, 256)

            segmentedArray.append(segArray)

        segmentedArray = np.stack(segmentedArray, axis=axis)
        print('SegmentedArray shape : {}'.format(segmentedArray.shape))

        segmented = sitk.GetImageFromArray(segmentedArray)
        saveImage(segmented, img, savePath)

    
        print('saving segmented label to {}...'.format(savepath), end='', flush=True)
 
    if axis == 2:
        length = originalShape[2]

        zero = np.zeros((256, 256)) - 1024.

        for x in range(length):
            if x == 0:
                3chArray = np.dstack([zero, resizedImageArray[:, :, x], resizedImageArray[:, :, x + 1]])

            elif x == (length - 1):
                3chArray = np.dstack([resizedImageArray[:, :, x - 1], resizedImageArray[:, :, x], zero])

            else:
                3chArray = np.dstack([resizedImageArray[:, :, x - 1], resizedImageArray[:, :, x], resizedImageArray[:, :, x + 1]])

            
            3chArray = 3chArray[np.newaxis,...]
            print('Shape of input shape: {}'.format(3chArray.shape))
            
            print('segmenting...')
            segArray = model.predict(3chArray, batch_size=args.batchsize, verbose=0)
            segArray = np.argmax(segArray, axis=-1).astype(np.uint8)
            segArray = segArray.reshape(256, 256)

            segmentedArray.append(segArray)

        segmentedArray = np.stack(segmentedArray, axis=axis)
        print('SegmentedArray shape : {}'.format(segmentedArray.shape))

        segmented = sitk.GetImageFromArray(segmentedArray)
        saveImage(segmented, img, savePath)

    
        print('saving segmented label to {}...'.format(savepath), end='', flush=True)
 
if __name__ == '__main__':
    args = ParseArgs()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]])
