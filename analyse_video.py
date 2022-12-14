import math
import os
import mlflow
from func_timeout import func_set_timeout
from moviepy.editor import VideoFileClip
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import time


class video_yoga_analyser:
    def __init__(self, confidence_threshold, rf_model_path, nn_model_path):
        self.label_names = {
            0: "Akarna Dhanurasana",
            1: "Bharadvaja's Twist Pose (Bharadvajasana I)",
            2: "Boat Pose (Paripurna Navasana)",
            3: "Bound Angle Pose (Baddha Konasana)",
            4: "Bow Pose (Dhanurasana)",
            5: "Bridge Pose (Setu Bandha Sarvangasana)",
            6: "Camel Pose (Ustrasana)",
            7: "Cat Cow Pose (Marjaryasana)",
            8: "Chair Pose (Utkatasana)",
            9: "Child Pose (Balasana)",
            10: "Cobra Pose (Bhujangasana)",
            11: "Cockerel Pose",
            12: "Corpse Pose (Savasana)",
            13: "Cow Face Pose (Gomukhasana)",
            14: "Crane (Crow) Pose (Bakasana)",
            15: "Dolphin Plank Pose (Makara Adho Mukha Svanasana)",
            16: "Dolphin Pose (Ardha Pincha Mayurasana)",
            17: "Downward-Facing Dog Pose (Adho Mukha Svanasana)",
            18: "Eagle Pose (Garudasana)",
            19: "Eight-Angle Pose (Astavakrasana)",
            20: "Extended Puppy Pose (Uttana Shishosana)",
            21: "Extended Revolved Side Angle Pose (Utthita Parsvakonasana)",
            22: "Extended Revolved Triangle Pose (Utthita Trikonasana)",
            23: "Feathered Peacock Pose (Pincha Mayurasana)",
            24: "Firefly Pose (Tittibhasana)",
            25: "Fish Pose (Matsyasana)",
            26: "Four-Limbed Staff Pose (Chaturanga Dandasana)",
            27: "Frog Pose (Bhekasana)",
            28: "Garland Pose (Malasana)",
            29: "Gate Pose (Parighasana)",
            30: "Half Lord of the Fishes Pose (Ardha Matsyendrasana)",
            31: "Half Moon Pose (Ardha Chandrasana)",
            32: "Handstand Pose (Adho Mukha Vrksasana)",
            33: "Happy Baby Pose (Ananda Balasana)",
            34: "Head-to-Knee Forward Bend Pose (Janu Sirsasana)",
            35: "Heron Pose (Krounchasana)",
            36: "Intense Side Stretch Pose (Parsvottanasana)",
            37: "Legs-Up-the-Wall Pose (Viparita Karani)",
            38: "Locust Pose (Salabhasana)",
            39: "Lord of the Dance Pose (Natarajasana)",
            40: "Low Lunge Pose (Anjaneyasana)",
            41: "Noose Pose (Pasasana)",
            42: "Peacock Pose (Mayurasana)",
            43: "Pigeon Pose (Kapotasana)",
            44: "Plank Pose (Kumbhakasana)",
            45: "Plow Pose (Halasana)",
            46: "Pose Dedicated to the Sage Koundinya (Eka Pada Koundinyanasana I, II)",
            47: "Rajakapotasana",
            48: "Reclining Hand-to-Big-Toe Pose (Supta Padangusthasana)",
            49: "Revolved Head-to-Knee Pose (Parivrtta Janu Sirsasana)",
            50: "Scale Pose (Tolasana)",
            51: "Scorpion Pose (Vrischikasana)",
            52: "Seated Forward Bend Pose (Paschimottanasana)",
            53: "Shoulder-Pressing Pose (Bhujapidasana)",
            54: "Side-Reclining Leg Lift Pose (Anantasana)",
            55: "Side Crane (Crow) Pose (Parsva Bakasana)",
            56: "Side Plank Pose (Vasisthasana)",
            57: "Sitting Pose 1 (normal)",
            58: "Split Pose",
            59: "Staff Pose (Dandasana)",
            60: "Standing Forward Bend Pose (Uttanasana)",
            61: "Standing Split Pose (Urdhva Prasarita Eka Padasana)",
            62: "Standing Big Toe Hold Pose (Utthita Padangusthasana)",
            63: "Supported Headstand Pose (Salamba Sirsasana)",
            64: "Supported Shoulderstand Pose (Salamba Sarvangasana)",
            65: "Supta Baddha Konasana",
            66: "Supta Virasana Vajrasana",
            67: "Tortoise Pose",
            68: "Tree Pose (Vrksasana)",
            69: "Upward Bow (Wheel) Pose (Urdhva Dhanurasana)",
            70: "Upward Facing Two-Foot Staff Pose (Dwi Pada Viparita Dandasana)",
            71: "Upward Plank Pose (Purvottanasana)",
            72: "Virasana (Vajrasana)",
            73: "Warrior III Pose (Virabhadrasana III)",
            74: "Warrior II Pose (Virabhadrasana II)",
            75: "Warrior I Pose (Virabhadrasana I)",
            76: "Wide-Angle Seated Forward Bend Pose (Upavistha Konasana)",
            77: "Wide-Legged Forward Bend Pose (Prasarita Padottanasana)",
            78: "Wild Thing Pose (Camatkarasana)",
            79: "Wind Relieving Pose (Pawanmuktasana)",
            80: "Yogic Sleep Pose",
            81: "Reverse Warrior Pose (Viparita Virabhadrasana)"
        }

        self.keypoint_idx_pairs = [(0, 1), (0, 2), (0, 5), (0, 6), (1, 3), (2, 4), (5, 6), (5, 7), (5, 11), (6, 8),
                                   (6, 12), (7, 9),
                                   (8, 10), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]

        self.check_map = {0: [10, 11],
                          1: [0, 1, 8, 9, 10, 11],
                          2: [0, 1, 10, 11],
                          3: [0, 1, 8, 9, 10, 11],
                          4: [0, 1, 8, 9, 10, 11],
                          5: [8, 9, 10, 11],
                          6: [0, 1, 8, 9],
                          7: [0, 1, 8, 9, 10, 11],
                          8: [0, 1, 8, 9, 10, 11],
                          9: [0, 1, 8, 9, 10, 11],
                          10: [2, 3, 10, 11],
                          11: [0, 1, 8, 9, 10, 11],
                          12: [0, 1, 2, 3, 4, 5],
                          13: [6, 7, 8, 9],
                          14: [8, 9, 10, 11],
                          15: [2, 3, 6, 7],
                          16: [2, 3, 6, 7, 10, 11],
                          17: [0, 1, 10, 11],
                          18: [6, 7],
                          19: [6, 7, 10, 11],
                          20: [0, 1, 8, 9, 10, 11],
                          21: [],
                          22: [0, 1, 2, 3, 10, 11],
                          23: [6, 7],
                          24: [0, 1],
                          25: [2, 3, 6, 7],
                          26: [2, 3, 4, 5, 6, 7],
                          27: [6, 7, 8, 9, 10, 11],
                          28: [6, 7, 8, 9, 10, 11],
                          29: [0, 1],
                          30: [10, 11],
                          31: [0, 1, 2, 3],
                          32: [0, 1, 2, 3, 4, 5],
                          33: [0, 1, 8, 9, 10, 11],
                          34: [10, 11],
                          35: [0, 1, 10, 11],
                          36: [2, 3, 10, 11],
                          37: [0, 1, 2, 3, 10, 11],
                          38: [0, 1, 2, 3],
                          39: [0, 1, 10, 11],
                          40: [0, 1, 10, 11],
                          41: [6, 7, 8, 9, 10, 11],
                          42: [2, 3, 6, 7],
                          43: [10, 11],
                          44: [2, 3, 4, 5],
                          45: [0, 1, 2, 3, 10, 11],
                          46: [2, 3, 6, 7],
                          47: [6, 7, 8, 9, 10, 11],
                          48: [0, 1, 2, 3],
                          49: [10, 11],
                          50: [0, 1, 8, 9, 10, 11],
                          51: [6, 7, 8, 9, 10, 11],
                          52: [2, 3, 10, 11],
                          53: [0, 1, 8, 9, 10, 11],
                          54: [2, 3],
                          55: [8, 9, 10, 11],
                          56: [0, 1, 2, 3, 4, 5],
                          57: [8, 9, 10, 11],
                          58: [2, 3, 10, 11],
                          59: [0, 1, 2, 3, 10, 11],
                          60: [2, 3, 10, 11],
                          61: [2, 3],
                          62: [2, 3],
                          63: [2, 3, 6, 7],
                          64: [2, 3, 6, 7],
                          65: [8, 9],
                          66: [4, 5, 8, 9],
                          67: [0, 1, 10, 11],
                          68: [6, 7],
                          69: [0, 1, 8, 9],
                          70: [2, 3, 6, 7, 10, 11],
                          71: [0, 1, 2, 3, 4, 5],
                          72: [8, 9, 10, 11],
                          73: [0, 1, 2, 3],
                          74: [0, 1],
                          75: [0, 1],
                          76: [2, 3, 10, 11],
                          77: [2, 3, 6, 7, 10, 11],
                          78: [0, 1, 10, 11],
                          79: [8, 9, 10, 11],
                          80: [10, 11],
                          81: [0, 1]}

        self.confidence_threshold = confidence_threshold
        self.nn_model = tf.keras.models.load_model(nn_model_path)
        self.rf_model = mlflow.sklearn.load_model(rf_model_path)
        self.class_confidence=0.8

    def process_keypoints_normalized_square(self, keypoints):
        right_hip_y = keypoints[11 * 2 + 0]
        right_hip_x = keypoints[11 * 2 + 1]
        left_hip_y = keypoints[12 * 2 + 0]
        left_hip_x = keypoints[12 * 2 + 1]
        middle_point_y = (right_hip_y + left_hip_y) / 2.0
        middle_point_x = (right_hip_x + left_hip_x) / 2.0

        max_distance = 0.0

        for i in range(0, 17):
            y = keypoints[i * 2 + 0]
            x = keypoints[i * 2 + 1]
            distance = math.sqrt(math.pow((y - middle_point_y), 2) + math.pow((x - middle_point_x), 2))
            if distance > max_distance:
                max_distance = distance

        min_x = middle_point_x - max_distance
        max_x = middle_point_x + max_distance
        min_y = middle_point_y - max_distance
        max_y = middle_point_y + max_distance

        for i in range(0, 17):
            keypoints[i * 2 + 1] = keypoints[i * 2 + 1] - min_x
            keypoints[i * 2 + 0] = keypoints[i * 2 + 0] - min_y

        x_len = max_x - min_x
        y_len = max_y - min_y

        x_mul = 1.0 / x_len
        y_mul = 1.0 / y_len

        for i in range(0, 17):
            keypoints[i * 2 + 1] = keypoints[i * 2 + 1] * x_mul
            keypoints[i * 2 + 0] = keypoints[i * 2 + 0] * y_mul

        return keypoints

    @func_set_timeout(20)
    def calculate_points(self, frame):
        # lightning: 192, thunder: 256
        input_size = 256
        keypoints_with_scores = []
        input_image = tf.expand_dims(frame, axis=0)
        input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

        try:
            input_image = tf.cast(input_image, dtype=tf.int32)
            outputs = self.movenet_model(input_image)
            keypoints_with_scores = outputs['output_0'].numpy()[0][0]
            keypoints_with_scores = keypoints_with_scores.flatten().tolist()
        except:
            print("Point calculation timed out")

        return keypoints_with_scores

    def calculate_keypoints(self, video):
        frame_keypoints = []
        good_frame_idx = []
        current_frame_idx = 0
        for frame in video:
            height, width, channel = frame.shape
            converted_img = tf.expand_dims(frame, axis=0)
            converted_img = tf.image.convert_image_dtype(converted_img, tf.float32)
            result = self.detector(converted_img)
            names = [tf.compat.as_str_any(tensor.numpy()) for tensor in result["detection_class_entities"]]
            box_scores = result["detection_scores"].numpy()
            [y_box_min, x_box_min, y_box_max, x_box_max] = self.crop_box
            current_best_box_keypoints = []

            for idx in range(0, len(names)):
                if names[idx] in ['Man', 'Woman', 'Person', 'Girl', 'Boy']:
                    box_score = box_scores[idx]
                    if box_score >= 0.1:
                        [y_cb_min, x_cb_min, y_cb_max, x_cb_max] = result["detection_boxes"][idx].numpy()
                        y_cb_min = int(0.9 * y_cb_min * height)
                        y_cb_max = int(min(1.1 * y_cb_max, 1.0) * height)
                        x_cb_min = int(0.9 * x_cb_min * width)
                        x_cb_max = int(min(1.1 * x_cb_max, 1.0) * width)

                        if len(current_best_box_keypoints) > 0:
                            [y_best_box_min, x_best_box_min, y_best_box_max,
                             x_best_box_max] = current_best_box_keypoints
                            if y_cb_min < y_best_box_min:
                                y_best_box_min = y_cb_min
                            if y_cb_max > y_best_box_max:
                                y_best_box_max = y_cb_max
                            if x_cb_min < x_best_box_min:
                                x_best_box_min = x_cb_min
                            if x_cb_max > x_best_box_max:
                                x_best_box_max = x_cb_max
                            current_best_box_keypoints = [int(y_best_box_min), int(x_best_box_min),
                                                          int(y_best_box_max),
                                                          int(x_best_box_max)]
                        else:
                            current_best_box_keypoints = [int(y_cb_min), int(x_cb_min), int(y_cb_max),
                                                          int(x_cb_max)]

            if len(current_best_box_keypoints) > 0:
                self.crop_box = current_best_box_keypoints
                [y_box_min, x_box_min, y_box_max, x_box_max] = self.crop_box
            last_k_x = [k[0] for k in self.last_keypoint]
            last_k_y = [k[1] for k in self.last_keypoint]

            if len(self.last_keypoint) > 0:
                y_kb_min = min(last_k_y) * 0.9 / self.size_multiplier
                y_kb_max = min(max(last_k_y) * 1.1 / self.size_multiplier, self.new_height)
                x_kb_min = min(last_k_x) * 0.9 / self.size_multiplier
                x_kb_max = min(max(last_k_x) * 1.1 / self.size_multiplier, self.new_width)

                if y_kb_min < y_box_min:
                    y_box_min = y_kb_min
                if y_kb_max > y_box_max:
                    y_box_max = y_kb_max
                if x_kb_min < x_box_min:
                    x_box_min = x_kb_min
                if x_kb_max > x_box_max:
                    x_box_max = x_kb_max
                self.crop_box = [int(y_box_min), int(x_box_min), int(y_box_max), int(x_box_max)]

            [y_box_min, x_box_min, y_box_max, x_box_max] = self.crop_box

            self.boxes.append([int(y_box_min * self.size_multiplier),
                               int(x_box_min * self.size_multiplier),
                               int(y_box_max * self.size_multiplier),
                               int(x_box_max * self.size_multiplier)])
            box_height = y_box_max - y_box_min
            box_width = x_box_max - x_box_min

            frame = tf.image.crop_to_bounding_box(frame, y_box_min, x_box_min, box_height, box_width)

            keypoints_with_scores = self.calculate_points(frame)
            if len(keypoints_with_scores) > 0:
                current_keypoints_for_animation = []
                x_current_keypoints = []
                percents = []
                # [0:y, 1:x]
                plus_y = 0
                plus_x = 0

                bigger_side = box_height

                if box_height > box_width:
                    plus_x = box_height - box_width
                    plus_x = plus_x / 2
                else:
                    plus_y = box_width - box_height
                    plus_y = plus_y / 2
                    bigger_side = box_width

                for i in range(0, 17):
                    x_current_keypoints.append(float(keypoints_with_scores[i * 3]))
                    x_current_keypoints.append(float(keypoints_with_scores[i * 3 + 1]))
                    percents.append(float(keypoints_with_scores[i * 3 + 2]))
                    point_y = ((keypoints_with_scores[i * 3] * bigger_side - plus_y) + y_box_min) * self.size_multiplier
                    if point_y < 0:
                        point_y = 0
                    if point_y > self.new_height * self.size_multiplier:
                        point_y = self.new_height * self.size_multiplier
                    point_x = ((keypoints_with_scores[
                                    i * 3 + 1] * bigger_side - plus_x) + x_box_min) * self.size_multiplier
                    if point_x < 0:
                        point_x = 0
                    if point_x > self.new_width * self.size_multiplier:
                        point_x = self.new_width * self.size_multiplier

                    current_keypoints_for_animation.append([int(point_x), int(point_y)])

                x_current_keypoints = self.process_keypoints_normalized_square(x_current_keypoints)
                if min(percents) > self.confidence_threshold:
                    frame_keypoints.append(x_current_keypoints)
                    self.keypoints_image_scale.append(current_keypoints_for_animation)
                    self.last_keypoint = current_keypoints_for_animation
                    good_frame_idx.append(current_frame_idx)
                else:
                    self.keypoints_image_scale.append([])
            else:
                self.keypoints_image_scale.append([])

            current_frame_idx += 1
        return frame_keypoints, good_frame_idx

    def calculate_poses(self, video, frame_keypoints, good_frame_idx):
        if frame_keypoints != []:
            frame_keypoints_nn = tf.convert_to_tensor(frame_keypoints)
            y_predicted_all_nn = self.nn_model.predict(frame_keypoints_nn)
            frame_keypoints_rf = np.array(frame_keypoints)
            y_predicted_all_rf = self.rf_model.predict_proba(frame_keypoints_rf)

            pa_idx = 0
            predicted_good_all_rf = []
            predicted_good_all_nn = []
            for i in range(0, len(video)):
                if i in good_frame_idx:
                    predicted_good_all_rf.append(y_predicted_all_rf[pa_idx])
                    predicted_good_all_nn.append(y_predicted_all_nn[pa_idx])
                    pa_idx += 1
                else:
                    predicted_good_all_rf.append([])
                    predicted_good_all_nn.append([])

            for idx in range(0, len(predicted_good_all_rf)):
                p_rf = predicted_good_all_rf[idx]
                p_nn = predicted_good_all_nn[idx]
                pose_added = False
                if len(p_nn) > 0:
                    best_five_idx = np.argsort(p_nn)[-5:]
                    best_five_idx = np.flip(best_five_idx)
                    self.best_five.append(best_five_idx)
                else:
                    self.best_five.append([])
                if len(p_nn) > 0:
                    if np.max(p_nn) > self.class_confidence:
                        y_predicted = np.argmax(p_nn)
                        self.poses.append(y_predicted)
                        pose_added = True
                if not pose_added:
                    if len(p_rf) > 0:
                        if np.max(p_rf) > self.class_confidence:
                            y_predicted = np.argmax(p_rf)
                            self.poses.append(y_predicted)
                            pose_added = True
                if not pose_added:
                    self.poses.append(-1)


        else:
            for i in range(0, len(video)):
                self.poses.append(-1)
                self.best_five.append([])

    def calculate_points_poses(self):
        print('Start calculating keypoints and poses...')
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        model_name = "movenet_thunder"

        self.movenet_model = module.signatures['serving_default']
        start_time = 0
        end_time = 3
        idx = 0
        self.last_keypoint = []
        with VideoFileClip(self.fps_video_path) as clip:
            duration = clip.duration
            original_width, original_height = clip.size
            n_width = 640
            self.size_multiplier = 1.0
            if n_width < original_width:
                clip = clip.resize(width=n_width)
                self.size_multiplier = float(original_width) / n_width
            self.new_width, self.new_height = clip.size

            print('length:', clip.duration, 's, fps:', clip.fps, 'resized size:', clip.size)

            module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
            self.detector = hub.load(module_handle).signatures['default']

            self.crop_box = [0, 0, self.new_height, self.new_width]
            while start_time < duration:
                print('\t current time:', start_time, '-', end_time)
                sub_video_path = "temp" + str(idx) + ".gif"

                sub_clip = clip.subclip(start_time, end_time)
                sub_clip.write_videofile(sub_video_path, codec="gif", audio=False, logger=None)

                video = tf.io.read_file(sub_video_path)
                video = tf.image.decode_gif(video)

                frames_number, height, width, _ = video.shape
                frame_keypoints, good_frame_idx = self.calculate_keypoints(video)
                self.calculate_poses(video, frame_keypoints, good_frame_idx)

                start_time = start_time + 3
                end_time = end_time + 3
                if end_time > duration:
                    end_time = duration
                idx = idx + 1
                sub_clip.close()
                os.remove(sub_video_path)

        del clip
        print('Pose and keypoint calculation is done')

    def intersection(self, current_keypoint):
        if len(current_keypoint) == 17:
            right_shoulder = current_keypoint[6]
            left_shoulder = current_keypoint[5]
            right_hip = current_keypoint[12]
            left_hip = current_keypoint[11]

            right_v_x = right_shoulder[0] - right_hip[0]
            right_v_y = right_shoulder[1] - right_hip[1]
            left_v_x = left_shoulder[0] - left_hip[0]
            left_v_y = left_shoulder[1] - left_hip[1]
            if right_v_x != 0 and left_v_x != 0:
                right_m = right_v_y / right_v_x
                left_m = left_v_y / left_v_x
                if right_m != left_m:
                    x_intersection = (left_hip[1] - left_m * left_hip[0] - right_hip[1] + right_m * right_hip[0]) / (
                            right_m - left_m)
                    if (right_shoulder[0] <= x_intersection <= right_hip[0]) or (
                            right_shoulder[0] >= x_intersection >= right_hip[0]):
                        return True

        return False

    def fix_values(self):
        print('Start fixing values...')
        # check the no pose detected parts
        idx = 0
        while idx < len(self.poses):
            if self.poses[idx] == -1:
                if idx >= 1:
                    before_poses = self.poses[idx - 1: idx]
                    if len(before_poses) == 1:
                        if before_poses[0] != -1:
                            after_idx = idx
                            for a_idx in range(idx, len(self.poses)):
                                if self.poses[a_idx] != -1:
                                    after_idx = a_idx
                                    break
                            after_idx_end = len(self.poses) - 1
                            if after_idx < len(self.poses) - 1:
                                after_idx_end = after_idx + 1
                            after_poses = self.poses[after_idx:after_idx_end]
                            if len(after_poses) == 1:
                                if before_poses[0] == after_poses[0]:
                                    good_pose = after_poses[0]
                                    for i in range(idx, after_idx):
                                        if good_pose in self.best_five[i]:
                                            self.poses[i] = good_pose
                                    idx = after_idx - 1
            idx += 1

        # check the reccuring parts
        idx = 3
        while idx < len(self.poses) - 3:
            if self.poses[idx] != -1:
                current_pose = self.poses[idx]
                before_poses = self.poses[idx: idx + 3]
                if len(before_poses) == 3:
                    if before_poses[1] == current_pose and before_poses[2] == current_pose:
                        after_idx = idx
                        for a_idx in range(idx + 3, idx + 13):
                            if a_idx < len(self.poses) - 3:
                                if self.poses[a_idx] == current_pose:
                                    prob_after_poses = self.poses[a_idx: a_idx + 3]
                                    if prob_after_poses[1] == current_pose and prob_after_poses[2] == current_pose:
                                        after_idx = a_idx
                                        break
                        after_idx_end = len(self.poses)
                        if after_idx < len(self.poses) - 3:
                            after_idx_end = after_idx + 3
                        if after_idx_end > idx + 3:
                            after_poses = self.poses[after_idx:after_idx_end]
                            if len(after_poses) == 3:
                                if after_poses[0] == current_pose and after_poses[1] == current_pose and after_poses[2] == current_pose:
                                    for i in range(idx, after_idx_end):
                                        self.poses[i] = current_pose
                                    idx = after_idx - 1
            idx += 1

        # check the poses that are only for a small amount of time
        last_pose = None
        pose_counter = 0
        for idx in range(0, len(self.poses)):
            if self.poses[idx] != last_pose:
                if last_pose is not None:
                    if pose_counter <= 5:
                        for i in range(max(0, idx - pose_counter), idx):
                            self.poses[i] = -1
                    pose_counter = 1
                    last_pose = self.poses[idx]
                else:
                    pose_counter = 1
                    last_pose = self.poses[idx]
            else:
                pose_counter += 1
                last_pose = self.poses[idx]
            if idx == len(self.poses) - 1:
                if pose_counter <= 5:
                    for i in range(max(0, idx - pose_counter), idx + 1):
                        self.poses[i] = -1

        idx = 0
        while idx < len(self.poses):
            if self.poses[idx] == -1:
                if idx >= 3:
                    before_poses = self.poses[idx - 3: idx]
                    if len(before_poses) == 3:
                        if before_poses[0] != -1:
                            if before_poses[0] == before_poses[1] and before_poses[0] == before_poses[2]:
                                after_idx = idx
                                for a_idx in range(idx, len(self.poses)):
                                    if self.poses[a_idx] != -1:
                                        after_idx = a_idx
                                        break
                                after_idx_end = len(self.poses) - 1
                                if after_idx < len(self.poses) - 3:
                                    after_idx_end = after_idx + 3
                                after_poses = self.poses[after_idx:after_idx_end]
                                if len(after_poses) == 3:
                                    if before_poses[0] == after_poses[0]:
                                        if after_poses[0] == after_poses[1] and after_poses[0] == after_poses[2]:
                                            good_pose = after_poses[0]
                                            for i in range(idx, after_idx):
                                                self.poses[i] = good_pose
                                            idx = after_idx - 1
            idx += 1

        # fix the missing keypoints
        before_keypoint = None
        for idx in range(0, len(self.keypoints_image_scale)):
            if len(self.keypoints_image_scale[idx]) == 0:
                if before_keypoint is not None:
                    next_idx = -1
                    for n_idx in range(idx, len(self.keypoints_image_scale)):
                        if len(self.keypoints_image_scale[n_idx]) == 17:
                            next_idx = n_idx
                            break
                    if idx < next_idx < idx + 2:
                        after_keypoint = self.keypoints_image_scale[next_idx]
                        for i in range(idx, next_idx):
                            self.keypoints_image_scale[i] = after_keypoint
            else:
                before_keypoint = self.keypoints_image_scale[idx]

        # fix the left and right hip and shoulders if needed
        for idx in range(1, len(self.keypoints_image_scale) - 1):
            if (self.intersection(self.keypoints_image_scale[idx])):
                if (not self.intersection(self.keypoints_image_scale[idx - 1]) and not self.intersection(
                        self.keypoints_image_scale[idx + 1])):
                    temp_kp = self.keypoints_image_scale[idx][6]
                    self.keypoints_image_scale[idx][6] = self.keypoints_image_scale[idx][5]
                    self.keypoints_image_scale[idx][5] = temp_kp

        print('Values are fixed')

    def degree_between_points(self, a, b, c):
        [a_x, a_y] = a
        [b_x, b_y] = b
        [c_x, c_y] = c
        a_x_b_x = math.pow((a_x - b_x), 2)
        a_x_c_x = math.pow((a_x - c_x), 2)
        b_x_c_x = math.pow((b_x - c_x), 2)

        a_y_b_y = math.pow((a_y - b_y), 2)
        a_y_c_y = math.pow((a_y - c_y), 2)
        b_y_c_y = math.pow((b_y - c_y), 2)

        cos_angle = np.arccos(
            (a_x_b_x + a_y_b_y + b_x_c_x + b_y_c_y - a_x_c_x - a_y_c_y)
            / (2 * math.sqrt(a_x_b_x + a_y_b_y) * math.sqrt(b_x_c_x + b_y_c_y)))

        return np.degrees(cos_angle)

    def decide_color(self, current_keypoints, current_pose):
        line_color = (229, 204, 255)
        bad_color=(102,0,204)
        color = line_color

        if current_pose in self.check_map.keys():
            check_list = self.check_map[current_pose]
            # right arm straight
            if 0 in check_list:
                right_arm_deg = self.degree_between_points(current_keypoints[6], current_keypoints[8],
                                                           current_keypoints[10])
                if 150 <= right_arm_deg <= 210:
                    color = line_color
                else:
                    color = bad_color
                    return color
            # left arm straight
            if 1 in check_list:
                left_arm_deg = self.degree_between_points(current_keypoints[5], current_keypoints[7],
                                                          current_keypoints[9])
                if 150 <= left_arm_deg <= 210:
                    color = line_color
                else:
                    color = bad_color
                    return color
            # right leg straight
            if 2 in check_list:
                right_leg_deg = self.degree_between_points(current_keypoints[12], current_keypoints[14],
                                                           current_keypoints[16])
                if 150 <= right_leg_deg <= 210:
                    color = line_color
                else:
                    color = bad_color
                    return color
            # left leg straight
            if 3 in check_list:
                left_leg_deg = self.degree_between_points(current_keypoints[11], current_keypoints[13],
                                                          current_keypoints[15])
                if 150 <= left_leg_deg <= 210:
                    color = line_color
                else:
                    color = bad_color
                    return color
            # right half body straight
            if 4 in check_list:
                right_half_body_deg = self.degree_between_points(current_keypoints[6], current_keypoints[12],
                                                                 current_keypoints[14])
                if 150 <= right_half_body_deg <= 210:
                    color = line_color
                else:
                    color = bad_color
                    return color
            # left half body straight
            if 5 in check_list:
                left_half_body_deg = self.degree_between_points(current_keypoints[5], current_keypoints[11],
                                                                current_keypoints[13])
                if 150 <= left_half_body_deg <= 210:
                    color = line_color
                else:
                    color = bad_color
                    return color

            # right arm bent
            if 6 in check_list:
                right_arm_deg = self.degree_between_points(current_keypoints[6], current_keypoints[8],
                                                           current_keypoints[10])
                if 160 > right_arm_deg or right_arm_deg > 200:
                    color = line_color
                else:
                    color = bad_color
                    return color
            # left arm bent
            if 7 in check_list:
                left_arm_deg = self.degree_between_points(current_keypoints[5], current_keypoints[7],
                                                          current_keypoints[9])
                if 160 > left_arm_deg or left_arm_deg > 200:
                    color = line_color
                else:
                    color = bad_color
                    return color
            # right leg bent
            if 8 in check_list:
                right_leg_deg = self.degree_between_points(current_keypoints[12], current_keypoints[14],
                                                           current_keypoints[16])
                if 160 > right_leg_deg or right_leg_deg > 200:
                    color = line_color
                else:
                    color = bad_color
                    return color
            # left leg bent
            if 9 in check_list:
                left_leg_deg = self.degree_between_points(current_keypoints[11], current_keypoints[13],
                                                          current_keypoints[15])
                if 160 > left_leg_deg or left_leg_deg > 200:
                    color = line_color
                else:
                    color = bad_color
                    return color
            # right half body bent
            if 10 in check_list:
                right_half_body_deg = self.degree_between_points(current_keypoints[6], current_keypoints[12],
                                                                 current_keypoints[14])
                if 160 > right_half_body_deg or right_half_body_deg > 200:
                    color = line_color
                else:
                    color = bad_color
                    return color
            # left half body bent
            if 11 in check_list:
                left_half_body_deg = self.degree_between_points(current_keypoints[5], current_keypoints[11],
                                                                current_keypoints[13])
                if 160 > left_half_body_deg or left_half_body_deg > 200:
                    color = line_color
                else:
                    color = bad_color
                    return color

        return color

    def animate_video(self):
        print('Start video animation...')
        analysed_video_name = self.result_directory_path + '.'.join(self.video_parts[0:-1]) + '_yoga_result.avi'

        final_video_name = self.result_directory_path + '.'.join(self.video_parts[0:-1]) + '_result.mp4'

        # colors: gbr
        text_color = (155, 0, 155)
        point_color=(153,51,255)

        cap = cv2.VideoCapture(self.fps_video_path)
        pose_idx = 0
        point_idx = 0
        box_idx = 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        font = cv2.FONT_HERSHEY_SIMPLEX

        # source: https://stackoverflow.com/questions/52846474/how-to-resize-text-for-cv2-puttext-according-to-the-image-size-in-opencv-python
        FONT_SCALE = 0.001
        THICKNESS_SCALE = 0.002
        font_scale = min(frame_width, frame_height) * FONT_SCALE
        thickness = math.ceil(min(frame_width, frame_height) * THICKNESS_SCALE)
        before_thickness = max(1, int(0.6 * thickness))
        point_size = int(5 * font_scale)

        max_text_height = 0
        max_text_width = 0
        for p in self.poses:
            if (p >= 0):
                p_max_text = self.label_names[p] + ' 1000 s'
                text_height = \
                    cv2.getTextSize(p_max_text, fontFace=font, fontScale=font_scale, thickness=thickness)[0][1]
                text_width = \
                    cv2.getTextSize(p_max_text, fontFace=font, fontScale=font_scale, thickness=thickness)[0][0]
                if text_height > max_text_height:
                    max_text_height = text_height
                if text_width > max_text_width:
                    max_text_width = text_width

        text_place = (10, 10 + max_text_height)

        size = (frame_width, frame_height)

        result = cv2.VideoWriter(analysed_video_name,
                                 cv2.VideoWriter_fourcc(*'DIVX'),
                                 fps, size)

        all_poses = []
        last_pose = -1
        pose_time_counter = 0

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                currentKeypoints = self.keypoints_image_scale[point_idx]
                current_pose = self.poses[pose_idx]
                currentBox = self.boxes[box_idx]
                frame_cpy = frame.copy()
                before_text_heights = 0
                for (p, time) in all_poses[-5:]:
                    before_text_heights += \
                        cv2.getTextSize(self.label_names[p], fontFace=font, fontScale=font_scale, thickness=thickness)[
                            0][1]
                cv2.rectangle(frame, (0, 0), (max_text_width + 20, max_text_height + before_text_heights + 20),
                              (255, 255, 255), -1)
                cv2.rectangle(frame, (currentBox[1], currentBox[0]),
                              (currentBox[3], currentBox[2]), (255, 255, 255), point_size)
                alpha = 0.3
                frame_overlay = cv2.addWeighted(frame, alpha, frame_cpy, 1 - alpha, gamma=0)

                if len(currentKeypoints) > 1:
                    color = self.decide_color(currentKeypoints, current_pose)
                    for (keypoint_idx_one, keypoint_idx_two) in self.keypoint_idx_pairs:
                        if (len(currentKeypoints) > keypoint_idx_one) and (len(currentKeypoints) > keypoint_idx_two):
                            frame_overlay = cv2.line(frame_overlay, currentKeypoints[keypoint_idx_one],
                                                     currentKeypoints[keypoint_idx_two], color, point_size)

                for point in currentKeypoints:
                    if (len(point) > 0):
                        frame_overlay = cv2.circle(frame_overlay, point, 2 * point_size, point_color, -1)

                p_height_before = 0

                if len(all_poses[-5:]) >= 1:
                    rev_all_poses = all_poses[-5:][::-1]
                    for (p, time) in rev_all_poses:
                        all_pose_text = self.label_names[p] + ' ' + str(time) + ' s'
                        p_height = int(
                            cv2.getTextSize(all_pose_text, fontFace=font, fontScale=font_scale, thickness=thickness)[
                                0][1])
                        p_height_before += p_height
                        pose_place = (text_place[0], text_place[1] + p_height_before)
                        frame_overlay = cv2.putText(frame_overlay, all_pose_text, pose_place, font, 0.6 * font_scale,
                                                    text_color, before_thickness)

                if current_pose >= 0:
                    if current_pose != last_pose:
                        if pose_time_counter >= 5:
                            current_pose_time = pose_time_counter / 10
                            all_poses.append((last_pose, current_pose_time))
                        pose_time_counter = 0
                        last_pose = current_pose

                    if last_pose == current_pose:
                        pose_time_counter += 1

                    if pose_time_counter > 0:
                        current_pose_time = pose_time_counter / 10
                        pose_text = self.label_names[current_pose] + ' ' + str(current_pose_time) + ' s'
                        frame_overlay = cv2.putText(frame_overlay, pose_text, text_place,
                                                    font,
                                                    font_scale,
                                                    text_color,
                                                    thickness)
                    else:
                        pose_text = 'No pose detected'
                        frame_overlay = cv2.putText(frame_overlay, pose_text, text_place,
                                                    font,
                                                    font_scale,
                                                    text_color,
                                                    thickness)

                else:
                    if pose_time_counter >= 5:
                        current_pose_time = pose_time_counter / 10
                        all_poses.append((last_pose, current_pose_time))
                    pose_time_counter = 0
                    last_pose = current_pose

                    pose_text = 'No pose detected'
                    frame_overlay = cv2.putText(frame_overlay, pose_text, text_place,
                                                font,
                                                font_scale,
                                                text_color,
                                                thickness)

                if pose_idx < len(self.poses) - 1:
                    pose_idx += 1
                if point_idx < len(self.keypoints_image_scale) - 1:
                    point_idx += 1
                if box_idx < len(self.boxes) - 1:
                    box_idx += 1
                result.write(frame_overlay)
            else:
                break

        cap.release()
        del cap
        result.release()
        del result
        cv2.destroyAllWindows()

        # add audio back
        video_clip = VideoFileClip(analysed_video_name)
        audio_clip = VideoFileClip(self.fps_video_path)
        if audio_clip.audio is not None:
            video_clip.audio = audio_clip.audio
        video_clip.write_videofile(final_video_name, logger=None)
        audio_clip.close()
        video_clip.close()
        del video_clip
        del audio_clip
        os.remove(analysed_video_name)
        os.remove(self.fps_video_path)
        print('Video is animated')
        print('Result video path:', final_video_name)


    def analyse_video(self, video_path):
        print('Start analysing video: ', video_path)
        start_time = time.time()
        with VideoFileClip(video_path) as clip:
            file_path = video_path.split('/')[-1]
            self.video_parts = file_path.split('.')
            self.result_directory_path = './results/'
            fps_video_path = self.result_directory_path + '.'.join(self.video_parts[0:-1]) + '_10fps.mp4'
            clip = clip.set_fps(10)
            clip.write_videofile(fps_video_path, fps=10, audio=True, logger=None)
        del clip
        self.keypoints_image_scale = []
        self.poses = []
        self.boxes = []
        self.best_five = []
        self.fps_video_path = fps_video_path
        self.calculate_points_poses()
        self.fix_values()
        self.animate_video()
        end_time = time.time()
        print('Elapsed time:', end_time - start_time, 's')


def main():
    best_rf_model_path="./best_models/f5e56fb8bea84f9dbc493a3687ee565c/artifacts/model/"
    best_nn_model_path="./best_models/bd12973598dd4df793fb832b588d90b5/artifacts/model/data/model"
    print('YOGA VIDEO ANALYSER')
    print('Yoga video file name:')
    file_name = input()
    if os.path.isfile(file_name):
        analyser = video_yoga_analyser(0.1, best_rf_model_path, best_nn_model_path)
        analyser.analyse_video(file_name)
    else:
        print('File not found')


if __name__ == "__main__":
    main()
