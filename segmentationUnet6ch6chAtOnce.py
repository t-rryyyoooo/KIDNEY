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

    length = source.GetSize()[0]
    segmentedSize = [256, 256]
    sourceImageArrayList = []
    refImageArrayList = []
    for x in range(length):
        sourceImage = ResamplingInAxis(source, x, segmentedSize)
        refImage = ResamplingInAxis(ref, x, segmentedSize)
        sourceImageArray = sitk.GetArrayFromImage(sourceImage)
        refImageArray = sitk.GetArrayFromImage(refImage)

        sourceImageArrayList.append(sourceImageArray)
        refImageArrayList.append(refImageArray)

    zero = np.zeros(segmentedSize) - 1024
    segmentedArrayList = []

    for x in range(length):
        sourceMiddle = sourceImageArrayList[x]
        refMiddle = refImageArrayList[x] 
        if x == 0:
            sourceTop = zero
            refTop = zero
            sourceBottom = sourceImageArrayList[x + 1]
            refBottom = refImageArrayList[x + 1]

        elif x == (length - 1):
            sourceTop = sourceImageArrayList[x - 1]
            refTop = refImageArrayList[x - 1]
            sourceBottom = zero
            refBottom = zero

        else:
            sourceTop = sourceImageArrayList[x - 1]
            refTop = refImageArrayList[x - 1]
            sourceBottom = sourceImageArrayList[x + 1]
            refBottom = refImageArrayList[x + 1]

        souceStack = [sourceTop, sourceMiddle, sourceBottom]
        refStack = [refTop, refMiddle, refBottom]
        stack = sourceStack + refStack

        imageArray6ch = np.dstack(stack)
        imageArray6ch = imageArray6ch[np.newaxis, ...]
        print("Shape of input shape : {}".format(imageArray6ch.shape))
        print("Segmenting...")

        segmentedArray = model.predict(imageArray6ch, batch_size=args.batchsize, verbose=0)
        segmentedArray = np.argmax(segmentedArray, axis=-1).astype(np.int8)
        segmentedArray = segmentedArray.reshape(*newSize)

        segmentedArrayList.append(segmentedArray)
    
    segmentedArray = np.dstack(segmentedArrayList)
    segmented = sitk.GetImageFromArray(segmentedArray)

    dummy = ResampleSize(source, [length] + newSize)
    segmented.SetOrigin(dummy.GetOrigin())
    segmented.SetDirection(dummy.GetDirection())
    segmented.SetSpacing(dummy.GetSpacing())

    segmented = ResampleSize(segmented, source.GetSize(), is_label=True)

    print("segmentedArray shape : {}".format(segmented.GetSize()))

    sitk.WriteImage(segmented, savePath, True)


if __name__ == '__main__':
    args = ParseArgs()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]])
