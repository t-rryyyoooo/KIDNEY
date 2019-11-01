from PIL import Image
import glob
import os
import json
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
import numpy as np
import math
import random

"""
np.stackした後でデータを扱うか、
list型で扱うか考える必要がある。
またどの段階で変換するか。
可変性を残しつつ、最後はinput,teachをきれいに返すような形でmain.pyに送りたい。
今のところの課題
・メンバ変数
・inputのconcatnateをいつやるか（main.pyメソッドを呼び出して追加がいいかな？)（しかも、parserからIOできるように)

"""

config = json.load(open('./settings.json'))


class Loader(object):
    # コンストラクタ
    def __init__(self, json_paths, batch_size,init_size=(256, 256),parser=None):
        self.size = init_size
        self.DATASET_PATH = json_paths
        self.add_member()
        self.batch_size = batch_size
        self.parser = parser


    # def define_IO(self):

    def add_member(self):
        """
        jsonファイルに記載されている、pathをクラスメンバとして登録する。
        self.Left_RGBとかが追加されている。
        """
        for key, path in self.DATASET_PATH.items():
            setattr(self, key, glob.glob(os.path.join(path, '*png')))

    # 左右の画像を結合してデータを二倍にする
    def concat_left_right(self):
        self.Left_slide += self.Right_slide
        self.Left_RGB += self.Right_RGB
        self.Left_disparity += self.Left_disparity
        self.Right_disparity += self.Right_disparity
        self.Left_bin += self.Left_bin
        self.Right_bin += self.Right_bin
        print('Done concat_left_right.')

    # 入力で使う画像を平均値で埋めた画像にする
    def change_slide2fill(self):
        self.Left_slide = self.Left_fill
        self.Right_slide = self.Right_fill

    @staticmethod
    def img2normalize(img_array):
        img_array = img_array.astype(np.float16)
        img_array -= 122.5
        img_array /= 255
        return img_array

    @staticmethod
    def normalize2img(img_array):
        img_array = img_array.astype(np.float16)
        img_array *= 255
        img_array += 122.5
        img_array = img_array.astype(np.uint8)
        return img_array

    def return_gen(self):
        self.imgs_length = len(self.Left_RGB)
        # self.train_paths = (self.Left_slide, self.Right_disparity, self.Left_disparity)
        # sel = self.Left_RGB
        self.train_list, self.valid_list, self.test_list = self.train_valid_test_splits(self.imgs_length)
        self.train_steps = math.ceil(len(self.train_list) / self.batch_size)
        self.valid_steps = math.ceil(len(self.valid_list) / self.batch_size)
        self.test_steps = math.ceil(len(self.test_list) / self.batch_size)
        self.train_gen = self.generator_with_preprocessing(self.train_list, self.batch_size)
        self.valid_gen = self.generator_with_preprocessing(self.valid_list, self.batch_size)
        self.test_gen = self.generator_with_preprocessing(self.test_list, self.batch_size)
        return self.train_gen, self.valid_gen, self.test_gen

    def return_step(self):
        return self.train_steps, self.valid_steps, self.test_steps

    @staticmethod
    def train_valid_test_splits(imgs_length: 'int', train_rate=0.8, valid_rate=0.1, test_rate=0.1):
        data_array = list(range(imgs_length))
        tr = math.floor(imgs_length * train_rate)
        vl = math.floor(imgs_length * (train_rate + valid_rate))

        random.shuffle(data_array)
        train_list = data_array[:tr]
        valid_list = data_array[tr:vl]
        test_list = data_array[vl:]

        return train_list, valid_list, test_list

    # def load_img_no_generator(self,):

    def load_batch_img_array(self, batch_list, prepro_callback=False):
        teach_img_list = []
        input_img_list = []
        for i in batch_list:
            LS_img = img_to_array(
                load_img(self.Left_slide[i], color_mode='rgb', target_size=self.size)).astype(np.uint8)
            LD_img = img_to_array(
                load_img(self.Left_disparity[i], color_mode='grayscale', target_size=self.size)).astype(np.uint8)
            RD_img = img_to_array(
                load_img(self.Right_disparity[i], color_mode='grayscale', target_size=self.size)).astype(np.uint8)

            if self.parser.input_channel==7:
                LB_img = img_to_array(
                    load_img(self.Left_bin[i], color_mode='grayscale', target_size=self.size)).astype(np.uint8)
                # LB_img=np.where(LB_img==255,1,LB_img)

                RB_img = img_to_array(
                    load_img(self.Right_bin[i], color_mode='grayscale', target_size=self.size)).astype(np.uint8)
                # RB_img = np.where(RB_img == 255, 1, RB_img)

                input_img = np.concatenate((LS_img, LD_img, RD_img, LB_img, RB_img), 2).astype(np.uint8)
            else:
                input_img = np.concatenate((LS_img, LD_img, RD_img), 2).astype(np.uint8)

            teach_img = img_to_array(
                load_img(self.Left_RGB[i], color_mode='rgb', target_size=self.size)).astype(np.uint8)

            # バッチサイズが二倍になってしまう
            input_img_list.append(input_img)
            teach_img_list.append(teach_img)

        input_img_list = np.stack(input_img_list)
        teach_img_list = np.stack(teach_img_list)

        if self.parser.normalize_luminance:
            input_img_list=self.normalize2img(input_img_list)
            teach_img_list=self.normalize2img(teach_img_list)

        return input_img_list,teach_img_list

    def generator_with_preprocessing(self, img_id_list, batch_size):  # , *input_paths
        while True:
            for i in range(0, len(img_id_list), batch_size):
                batch_list = img_id_list[i:i + batch_size]
                batch_input, batch_teach = self.load_batch_img_array(batch_list)

                yield(batch_input, batch_teach)


class DataSet:
    pass
