import SimpleITK as sitk
import numpy as np
import argparse
import copy
import os
import sys
import copy
import tensorflow as tf
from functions import createParentPath, Resampling, cancer_dice, kidney_dice, penalty_categorical, saveImage
from clip3D import Resizing
from cut import sliceImage

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("sourcefile", help="imagefile")
    parser.add_argument("reffile", help="imagefile")
    parser.add_argument("modelweightfile", help="Trained model weights file (*.hdf5).")
    parser.add_argument("savepath", help="Segmented label file.(.mha)")
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

    savePath = os.path.expanduser(args.savepath)
    createParentPath(savePath)

    sourcefile = os.path.expanduser(args.sourcefile)
    reffile = os.path.expanduser(args.reffile)
    ## Read image
    source= sitk.ReadImage(sourcefile)
    ref= sitk.ReadImage(reffile)

    sourceArray = sitk.GetArrayFromImage(source)
    refArray = sitk.GetArrayFromImage(ref)

    print("Whole size: ",sourceArray.shape)

    originalShape = sourceArray.shape

    resizedSourceArray, axis = sliceImage(sourceArray, interpolation="linear")
    resizedRefArray, _ = sliceImage(refArray, interpolation="linear")
    
    segmentedArray = []
    if axis == 0:
        length = originalShape[0]

        zero = np.zeros((256, 256)) - 1024.

        for x in range(length):
            if x == 0:
                sourceStack = [zero, resizedSourceArray[x, :, :], resizedSourceArray[x + 1, :, :]]
                refStack = [zero, resizedRefArray[x, :, :], resizedRefArray[x + 1, :, :]]

            elif x == (length - 1):
                sourceStack = [resizedSourceArray[x - 1, :, :], resizedSourceArray[x, :, :], zero]
                refStack = [resizedRefArray[x - 1, :, :], resizedRefArray[x, :, :], zero]

            else:
                sourceStack = [resizedSourceArray[x - 1, :, :], resizedSourceArray[x, :, :], resizedSourceArray[x + 1, :, :]]
                refStack = [resizedRefArray[x - 1, :, :], resizedRefArray[x, :, :], resizedRefArray[x + 1, :, :]]

            
            stack = sourceStack + refStack
            imageArray3ch = np.dstack(stack)

            imageArray3ch = imageArray3ch[np.newaxis,...]
            print('Shape of input shape: {}'.format(imageArray3ch.shape))
            
            print('segmenting...')
            segArray = model.predict(imageArray3ch, batch_size=args.batchsize, verbose=0)
            segArray = np.argmax(segArray, axis=-1).astype(np.uint8)
            segArray = segArray.reshape(256, 256)

            segmentedArray.append(segArray)

        segmentedArray = np.stack(segmentedArray, axis=axis)
        dummyArray = np.zeros(originalShape)
        segmentedArray = Resizing(segmentedArray, dummyArray, interpolation="nearest")
        print('SegmentedArray shape : {}'.format(segmentedArray.shape))

        segmented = sitk.GetImageFromArray(segmentedArray)
        saveImage(segmented, source, savePath)
    
    if axis == 1:
        length = originalShape[1]

        zero = np.zeros((256, 256)) - 1024.

        for x in range(length):
            if x == 0:
                sourceStack = [zero, resizedSourceArray[:, x, :], resizedSourceArray[:, x + 1, :]]
                refStack = [zero, resizedRefArray[:, x, :], resizedRefArray[:, x + 1, :]]

            elif x == (length - 1):
                sourceStack = [resizedSourceArray[:, x - 1, :], resizedSourceArray[:, x, :], zero]
                refStack = [resizedRefArray[:, x - 1, :], resizedRefArray[:, x, :], zero]

            else:
                sourceStack = [resizedSourceArray[:, x - 1, :], resizedSourceArray[:, x, :], resizedSourceArray[:, x + 1, :]]
                refStack = [resizedRefArray[:, x - 1, :], resizedRefArray[:, x, :], resizedRefArray[:, x + 1, :]]

            
            stack = sourceStack + refStack
            imageArray3ch = np.dstack(stack)

            imageArray3ch = imageArray3ch[np.newaxis,...]
            print('Shape of input shape: {}'.format(imageArray3ch.shape))
            
            print('segmenting...')
            segArray = model.predict(imageArray3ch, batch_size=args.batchsize, verbose=0)
            segArray = np.argmax(segArray, axis=-1).astype(np.uint8)
            segArray = segArray.reshape(256, 256)

            segmentedArray.append(segArray)

        segmentedArray = np.stack(segmentedArray, axis=axis)
        dummyArray = np.zeros(originalShape)
        segmentedArray = Resizing(segmentedArray, dummyArray, interpolation="nearest")
        print('SegmentedArray shape : {}'.format(segmentedArray.shape))

        segmented = sitk.GetImageFromArray(segmentedArray)
        saveImage(segmented, source, savePath)
 

    if axis == 2:
        length = originalShape[2]

        zero = np.zeros((256, 256)) - 1024.

        for x in range(length):
            if x == 0:
                sourceStack = [zero, resizedSourceArray[:, :, x], resizedSourceArray[:, :, x + 1]]
                refStack = [zero, resizedRefArray[:, :, x], resizedRefArray[:, :, x + 1]]

            elif x == (length - 1):
                sourceStack = [resizedSourceArray[:, :, x - 1], resizedSourceArray[:, :, x], zero]
                refStack = [resizedRefArray[:, :, x - 1], resizedRefArray[:, :, x], zero]

            else:
                sourceStack = [resizedSourceArray[:, :, x - 1], resizedSourceArray[:, :, x], resizedSourceArray[:, :, x + 1]]
                refStack = [resizedRefArray[:, :, x - 1], resizedRefArray[:, :, x], resizedRefArray[:, :, x + 1]]

            
            stack = sourceStack + refStack
            imageArray3ch = np.dstack(stack)

            imageArray3ch = imageArray3ch[np.newaxis,...]
            print('Shape of input shape: {}'.format(imageArray3ch.shape))
            
            print('segmenting...')
            segArray = model.predict(imageArray3ch, batch_size=args.batchsize, verbose=0)
            segArray = np.argmax(segArray, axis=-1).astype(np.uint8)
            segArray = segArray.reshape(256, 256)

            segmentedArray.append(segArray)

        segmentedArray = np.stack(segmentedArray, axis=axis)
        dummyArray = np.zeros(originalShape)
        segmentedArray = Resizing(segmentedArray, dummyArray, interpolation="nearest")
        print('SegmentedArray shape : {}'.format(segmentedArray.shape))

        segmented = sitk.GetImageFromArray(segmentedArray)
        saveImage(segmented, source, savePath)
 
    
if __name__ == '__main__':
    args = ParseArgs()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]])
