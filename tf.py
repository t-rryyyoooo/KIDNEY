from tensorflow.python.client import device_lib
device_lib.list_local_devices()
import codecs
import tensorflow as tf

with tf.device('/device:GPU:0'):
        print('loading U-net model {}...'.format('/home/kakeya/Desktop/tanimoto/data/model/model_1.0.yml'), end='', flush=True)
        with open('/home/kakeya/Desktop/tanimoto/KIDNEY/testb.hdf5') as f:
            #model = tf.compat.v1.keras.models.model_from_yaml(f.read())
            new_model = tf.compat.v1.keras.models.load_model('/home/kakeya/Desktop/tanimoto/KIDNEY/test.yml')
        #model.load_weights('/home/kakeya/Desktop/tanimoto/KIDNEY/testb.hdf5')
        print('done')
