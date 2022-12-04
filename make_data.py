import csv

import numpy as np
import wget
import os
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import pandas as pd
from func_timeout import func_set_timeout
import time

YOGA_POSE_INDEX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                   28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                   53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                   78, 79, 80, 81]
input_size_dict = {"movenet_thunder": 256, "movenet_lightning": 192}
tfhub_model_dict = {"movenet_thunder": "https://tfhub.dev/google/movenet/singlepose/thunder/4",
                    "movenet_lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4"}


class keypoint_maker:
    def __init__(self, model_name):
        self.input_size = input_size_dict[model_name]
        tfhub_model = tfhub_model_dict[model_name]
        module = hub.load(tfhub_model)
        self.movenet_model = module.signatures['serving_default']

    @func_set_timeout(5)
    def movenet(self, input_image):
        input_image = tf.cast(input_image, dtype=tf.int32)
        print('Start model evaluation')
        outputs = self.movenet_model(input_image)
        print('Output is done')

        keypoints_with_scores = outputs['output_0'].numpy()[0][0]
        return keypoints_with_scores

    @func_set_timeout(10)
    def calculate_points(self, img_url):
        print('Image: ', img_url)
        keypoints_with_scores = []
        filename = ""
        try:
            filename = wget.download(img_url)
        except:
            print('URL not valid:', img_url)
            self.bad_url_counter += 1
            return keypoints_with_scores

        new_filename = 'temp_image.jpeg'
        try:
            with Image.open(filename) as im:
                im.save(new_filename, 'JPEG')
                image = tf.io.read_file(new_filename, 'rb')
                image = tf.image.decode_jpeg(image)
                input_image = tf.expand_dims(image, axis=0)
                input_image = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)
                keypoints_with_scores = self.movenet(input_image)
                os.remove(new_filename)
            os.remove(filename)
        except:
            print('Error in generating keypoints')
            self.movenet_error_counter += 1
            os.remove(filename)
        return keypoints_with_scores

    def make_dataset(self, data_path, save_path):
        self.bad_url_counter = 0
        self.movenet_error_counter = 0
        start_time = time.time()
        txt_names = {}
        temp_pose_ids = {}
        img_and_id_dict = {}
        with open(data_path, newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                [img_name, c_8, c_20, pose_id] = row
                pose_id = int(pose_id)
                if (pose_id in YOGA_POSE_INDEX):
                    [txt_name, img_no] = img_name.split('/')
                    if (txt_name not in txt_names.keys()):
                        txt_names.update({txt_name: []})
                        temp_pose_ids[txt_name] = pose_id
                    txt_names[txt_name].append(img_name)

        print('Reading txt files')
        for txt_name in txt_names.keys():
            with open('Yoga-82/yoga_dataset_links/' + txt_name + '.txt', newline='\n') as pose_file:
                print(txt_name)
                pose_reader = csv.reader(pose_file, delimiter='\t')
                for row in pose_reader:
                    [img_name, img_url] = row
                    if img_name in txt_names[txt_name]:
                        img_and_id_dict[img_url] = temp_pose_ids[txt_name]

        keypoints = []
        ids = []
        for img_url in img_and_id_dict.keys():
            keypoints_with_scores = []
            try:
                keypoints_with_scores = self.calculate_points(img_url)
            except:
                ('Function timed out')
            if (len(keypoints_with_scores) == 17):
                keypoints.append(keypoints_with_scores.flatten().tolist())
                print(keypoints_with_scores)
                ids.append(img_and_id_dict[img_url])

        data_info = {}
        for i in ids:
            if i not in data_info.keys():
                data_info[i] = 1
            else:
                data_info[i] += 1
        end_time = time.time()
        print('-------------------------DATA INFO-------------------------')
        print('Saved file:', save_path)
        print('Class id-number of data dictionary', data_info)
        print('URL not found:', self.bad_url_counter)
        print('Movenet keypoints not found:', self.movenet_error_counter)
        print('Data generation time:', end_time - start_time)
        print('-----------------------------------------------------------')

        rows = []
        print('Making dataframe')
        for i in range(0, len(ids)):
            row = keypoints[i]
            row.append(ids[i])
            rows.append(row)
        r = np.asarray(rows)
        pd.DataFrame(r).to_csv(save_path, header=False)

def main():
    print('Start generating data with MoveNet Thunder...')
    kp_maker_thunder = keypoint_maker("movenet_thunder")
    train_data_path = 'Yoga-82/yoga_train.txt'
    train_save_path = 'yoga_thunder_train.csv'
    print('Data path:', train_data_path)
    print('Saved file path:', train_save_path)
    kp_maker_thunder.make_dataset(train_data_path, train_save_path)
    print('\tTrain data saved')

    test_data_path = 'Yoga-82/yoga_test.txt'
    test_save_path = 'yoga_thunder_test.csv'
    print('Data path:', test_data_path)
    print('Saved file path:', test_save_path)
    kp_maker_thunder.make_dataset(test_data_path, test_save_path)
    print('\tTest data saved')

    print('Start generating data with MoveNet Lightning...')
    kp_maker_lightning = keypoint_maker("movenet_lightning")
    train_data_path = 'Yoga-82/yoga_train.txt'
    train_save_path = 'yoga_lightning_train.csv'
    print('Data path:', train_data_path)
    print('Saved file path:', train_save_path)
    kp_maker_lightning.make_dataset(train_data_path, train_save_path)
    print('\tTrain data saved')

    test_data_path = 'Yoga-82/yoga_test.txt'
    test_save_path = 'yoga_lightning_test.csv'
    print('Data path:', test_data_path)
    print('Saved file path:', test_save_path)
    kp_maker_lightning.make_dataset(test_data_path, test_save_path)
    print('\tTest data saved')

    print('All data is generated')

if __name__ == "__main__":
    main()

