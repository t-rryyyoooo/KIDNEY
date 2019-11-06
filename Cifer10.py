from tensorflow.python.keras.datasets import cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

print("x_train.shape: ",x_train.shape)
print("x_test.shapa: ",x_test.shape)
print("y_train.shape: ",y_train.shape)
print("y_test.shape: ",y_test.shape)

from tensorflow.python.keras.utils import to_categorical

x_train = x_train/255
x_test = x_test/255
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.callbacks import TensorBoard
import tensorflow as tf

with tf.device("/device:GPU:0"):
    model = Sequential()

    model.add(
        Conv2D(
            filters=32,
            input_shape=(32,32,3),
            kernel_size=(3,3),
            strides=(1,1),
            padding='same',
            activation="relu"
        )
    )

    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same',
            activation="relu"
        )
    )
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3,3),
            strides=(1,1),
            activation='relu'
        )
    )

    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same",
            activation="relu"
        )
    )

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(units=512,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=10,activation="softmax"))

    model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=['accuracy']
    )

    tsb = TensorBoard(log_dir="./log")
    history_model1 = model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=2,
        validation_split=0.2,
        callbacks=[tsb]
    )