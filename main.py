import argparse
import keras.callbacks
from PIL import Image, ImageOps
from IPython.display import display_png
import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from Models import deep_auto_encoder
from Models.simple_auto_encoder import Simple_auto_encoder
from Models.unet import UNet
from Models.dn_cnn import DN_CNN
import time

from tensorflow.python.keras.layers import Input
import tensorflow as tf
from datetime import datetime as dt
from tensorflow import keras
from Utils.reporter import Reporter
from Utils.loader import DataSet, Loader

import json
import glob

INPUT_SIZE = (256, 256)
SAVE_BATCH_SIZE=2
CONCAT_LEFT_RIGHT=True
CHANGE_SLIDE2_FILL = True



def train(parser):
    START_TIME = time.time()
    configs = json.load(open('./settings.json'))
    reporter = Reporter(parser=parser)
    loader = Loader(configs['dataset_path2'], parser.batch_size, parser=parser)
    
    if CHANGE_SLIDE2_FILL:
        loader.change_slide2fill()
        reporter.add_log_documents('Done change_slide2fill.')

    if CONCAT_LEFT_RIGHT:
        loader.concat_left_right()
        reporter.add_log_documents('Done concat_left_right.')


    train_gen, valid_gen, test_gen = loader.return_gen()
    train_steps, valid_steps, test_steps = loader.return_step()

    # ---------------------------model----------------------------------

    input_channel_count = parser.input_channel
    output_channel_count = 3
    first_layer_filter_count = parser.filter

    network = UNet(input_channel_count, output_channel_count, first_layer_filter_count,parser=parser)
    model = network.get_model()

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # ---------------------------training----------------------------------
    batch_size = parser.batch_size
    epochs = parser.epoch

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # fit_generatorのコールバック関数の指定・TensorBoardとEarlyStoppingの指定

    logdir = os.path.join('./logs', dt.today().strftime("%Y%m%d_%H%M"))
    os.makedirs(logdir, exist_ok=True)
    tb_cb = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=True)

    es_cb = EarlyStopping(monitor='val_loss', patience=parser.early_stopping, verbose=1, mode='auto')

    print("start training.")
    # Pythonジェネレータ（またはSequenceのインスタンス）によりバッチ毎に生成されたデータでモデルを訓練します．
    history = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=valid_gen,
        validation_steps=valid_steps,
        # use_multiprocessing=True,
        callbacks=[es_cb, tb_cb])

    print("finish training. And start making predict.")

    test_preds = model.predict_generator(test_gen, steps=test_steps, verbose=1)
   
    print("finish making predict. And render preds.")


    ELAPSED_TIME = int(time.time() - START_TIME)
    reporter.add_log_documents(f'ELAPSED_TIME:{ELAPSED_TIME} [sec]')
    
    # ==========================report====================================
    parser.save_logs=True
    if parser.save_logs:
        reporter.add_val_loss(history.history['val_loss'])
        reporter.add_model_name(network.__class__.__name__)
        reporter.generate_main_dir()
        reporter.plot_history(history)
        reporter.save_params(history)

        train_gen, valid_gen, _ = loader.return_gen()
        
        for i in range(min(train_steps,SAVE_BATCH_SIZE)):
            batch_input, batch_teach = next(train_gen)
            batch_preds = model.predict(batch_input)
            if parser.normalize_luminance:
                batch_input = loader.normalize2img(batch_input)
                batch_teach = loader.normalize2img(batch_teach)
            reporter.plot_predict2(batch_input,batch_preds,batch_teach,'train',batch_num=i)

        for i in range(min(valid_steps, SAVE_BATCH_SIZE)):
            batch_input, batch_teach = next(valid_gen)
            batch_preds = model.predict(batch_input)
            if parser.normalize_luminance:
                batch_input = loader.normalize2img(batch_input)
                batch_teach = loader.normalize2img(batch_teach)
            reporter.plot_predict2(batch_input, batch_preds, batch_teach, 'valid', batch_num=i)
            
        for i in range(min(test_steps, SAVE_BATCH_SIZE)):
            batch_input, batch_teach = next(test_gen)
            batch_preds = model.predict(batch_input)
            if parser.normalize_luminance:
                batch_input = loader.normalize2img(batch_input)
                batch_teach = loader.normalize2img(batch_teach)
            reporter.plot_predict2(batch_input, batch_preds, batch_teach, 'test', batch_num=i)


        model.save("model.h5")


def get_parser():
    parser = argparse.ArgumentParser(
        prog='generate parallax image using U-Net',
        usage='python main.py',
        description='This module　generate parallax image using U-Net.',
        add_help=True
    )
    
    parser.add_argument('-e', '--epoch', type=int,
                        default=200, help='Number of epochs')
    parser.add_argument('-f', '--filter', type=int,
                        default=64, help='Number of model first_filters')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=16, help='Batch size')
    parser.add_argument('-t', '--trainrate', type=float,
                        default=0.85, help='Training rate')
    parser.add_argument('-es', '--early_stopping', type=int,
                        default=10, help='early_stopping patience')

    parser.add_argument('-i', '--input_channel', type=int,
                        default=5, help='input_channel')
    parser.add_argument('-a', '--augmentation',
                        action='store_true', help='Number of epochs')
    parser.add_argument('-s', '--save_logs',
                        action='store_true', help='save or not logs')
    parser.add_argument('-in', '--insert_skip_inputs',
                        action='store_false', help='insert_skip_inputs')
    
    parser.add_argument('-n', '--normalize_luminance',
                        action='store_true', help='normalize_luminance')

    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    train(parser)
