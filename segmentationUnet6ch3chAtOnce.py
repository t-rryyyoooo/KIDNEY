import SimpleITK as sitk
import numpy as np
import argparse
import copy
import os
import sys
import copy
import tensorflow as tf
from functions import createParentPath, Resampling, cancer_dice, kidney_dice, penalty_categorical, saveImage, ResampleSize, ResamplingInAxis
from clip3D import Resizing
from cut import sliceImage

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile", help="imagefile")
    #parser.add_argument("modelfile", help="U-net model file (*.yml).")
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

    imagefile = os.path.expanduser(args.imagefile)
    ## Read image
    image = sitk.ReadImage(imagefile)

    print("Image shape : {}".format(image.GetSize()))
    length = image.GetSize()[0]
    segmentedSize = [256, 256]
    sliceImageArrayList = []
    for x in range(length):
        sliceImage = ResamplingInAxis(image, x, segmentedSize)
        sliceImageArray = sitk.GetArrayFromImage(sliceImage)
        sliceImageArrayList.append(sliceImageArray)
    
    zero = np.zeros(segmentedSize) - 1024.
    segmentedArrayList = []

    for x in range(length):
        middle = sliceImageArrayList[x]
        if x == 0:
            top = zero
            bottom = sliceImageArrayList[x + 1]

        elif x == (length - 1):
            top = sliceImageArrayList[x - 1]
            bottom = zero

        else:
            top = sliceImageArrayList[x - 1]
            bottom = sliceImageArrayList[x + 1]

        imageArray3ch = np.dstack([top, middle, bottom])
        imageArray3ch = imageArray3ch[np.newaxis, ...]
        print("Shape og input shape : {}".format(imageArray3ch.shape))
        print("Segmenting...")

        segmentedArray = model.predict(imageArray3ch, batch_size=args.batchsize, verbose=0)
        segmentedArray = np.argmax(segmentedArray, axis=-1).astype(np.int8)
        segmentedArray = segmentedArray.reshape(*segmentedSize)

        segmentedArrayList.append(segmentedArray)

    segmentedArray = np.dstack(segmentedArrayList)
    segmented = sitk.GetImageFromArray(segmentedArray)

    dummy = ResampleSize(image, [length] + segmentedSize)
    segmented.SetOrigin(dummy.GetOrigin())
    segmented.SetSpacing(dummy.GetSpacing())
    segmented.SetDirection(dummy.GetDirection())

    segmented = ResampleSize(segmented, image.GetSize(), is_label=True)
    
    print("SegmentedArray shape : {}".format(segmented.GetSize()))

    print("Saving image to {}...".format(savePath))

    sitk.WriteImage(segmented, savePath, True)


if __name__ == '__main__':
    args = ParseArgs()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]])
