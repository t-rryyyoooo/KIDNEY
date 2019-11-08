from tensorflow.python.client import device_lib
device_lib.list_local_devices()

import tensorflow as tf

with tf.device('/device:GPU:0'):
        print('loading U-net model {}...'.format('/home/kakeya/Desktop/tanimoto/data/model/model_1.0.yml'), end='', flush=True)
        with open('/home/kakeya/Downloads/2DUnetModel_re0.30.yml') as f:
            model = tf.compat.v1.keras.models.model_from_yaml(f.read())
        model.load_weights('/home/kakeya/Desktop/tanimoto/data/weight/best_1.0.hdf5')
        print('done')
