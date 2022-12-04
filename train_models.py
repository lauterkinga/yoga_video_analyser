import math

import pandas as pd
import tensorflow as tf
import csv
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import matplotlib.pyplot as plt
import mlflow.tensorflow
import mlflow.sklearn
import logging
import time
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score
import seaborn as sb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# https://www.tensorflow.org/guide/keras/train_and_evaluate

np.set_printoptions(threshold=sys.maxsize)


class yoga_pose_classification_trainer:
    def __init__(self, train_data_file, test_data_file, confidence_threshold=0.3):
        self.train_data_file = train_data_file
        self.test_data_file = test_data_file
        self.confidence_threshold = confidence_threshold
        self.label_names_82 = [
            "Akarna Dhanurasana",
            "Bharadvaja's Twist Pose (Bharadvajasana I)",
            "Boat Pose (Paripurna Navasana)",
            "Bound Angle Pose (Baddha Konasana)",
            "Bow Pose (Dhanurasana)",
            "Bridge Pose (Setu Bandha Sarvangasana)",
            "Camel Pose (Ustrasana)",
            "Cat Cow Pose (Marjaryasana)",
            "Chair Pose (Utkatasana)",
            "Child Pose (Balasana)",
            "Cobra Pose (Bhujangasana)",
            "Cockerel Pose",
            "Corpse Pose (Savasana)",
            "Cow Face Pose (Gomukhasana)",
            "Crane (Crow) Pose (Bakasana)",
            "Dolphin Plank Pose (Makara Adho Mukha Svanasana)",
            "Dolphin Pose (Ardha Pincha Mayurasana)",
            "Downward-Facing Dog Pose (Adho Mukha Svanasana)",
            "Eagle Pose (Garudasana)",
            "Eight-Angle Pose (Astavakrasana)",
            "Extended Puppy Pose (Uttana Shishosana)",
            "Extended Revolved Side Angle Pose (Utthita Parsvakonasana)",
            "Extended Revolved Triangle Pose (Utthita Trikonasana)",
            "Feathered Peacock Pose (Pincha Mayurasana)",
            "Firefly Pose (Tittibhasana)",
            "Fish Pose (Matsyasana)",
            "Four-Limbed Staff Pose (Chaturanga Dandasana)",
            "Frog Pose (Bhekasana)",
            "Garland Pose (Malasana)",
            "Gate Pose (Parighasana)",
            "Half Lord of the Fishes Pose (Ardha Matsyendrasana)",
            "Half Moon Pose (Ardha Chandrasana)",
            "Handstand Pose (Adho Mukha Vrksasana)",
            "Happy Baby Pose (Ananda Balasana)",
            "Head-to-Knee Forward Bend Pose (Janu Sirsasana)",
            "Heron Pose (Krounchasana)",
            "Intense Side Stretch Pose (Parsvottanasana)",
            "Legs-Up-the-Wall Pose (Viparita Karani)",
            "Locust Pose (Salabhasana)",
            "Lord of the Dance Pose (Natarajasana)",
            "Low Lunge Pose (Anjaneyasana)",
            "Noose Pose (Pasasana)",
            "Peacock Pose (Mayurasana)",
            "Pigeon Pose (Kapotasana)",
            "Plank Pose (Kumbhakasana)",
            "Plow Pose (Halasana)",
            "Pose Dedicated to the Sage Koundinya (Eka Pada Koundinyanasana I, II)",
            "Rajakapotasana",
            "Reclining Hand-to-Big-Toe Pose (Supta Padangusthasana)",
            "Revolved Head-to-Knee Pose (Parivrtta Janu Sirsasana)",
            "Scale Pose (Tolasana)",
            "Scorpion Pose (Vrischikasana)",
            "Seated Forward Bend Pose (Paschimottanasana)",
            "Shoulder-Pressing Pose (Bhujapidasana)",
            "Side-Reclining Leg Lift Pose (Anantasana)",
            "Side Crane (Crow) Pose (Parsva Bakasana)",
            "Side Plank Pose (Vasisthasana)",
            "Sitting Pose 1 (normal)",
            "Split Pose",
            "Staff Pose (Dandasana)",
            "Standing Forward Bend Pose (Uttanasana)",
            "Standing Split Pose (Urdhva Prasarita Eka Padasana)",
            "Standing Big Toe Hold Pose (Utthita Padangusthasana)",
            "Supported Headstand Pose (Salamba Sirsasana)",
            "Supported Shoulderstand Pose (Salamba Sarvangasana)",
            "Supta Baddha Konasana",
            "Supta Virasana Vajrasana",
            "Tortoise Pose",
            "Tree Pose (Vrksasana)",
            "Upward Bow (Wheel) Pose (Urdhva Dhanurasana)",
            "Upward Facing Two-Foot Staff Pose (Dwi Pada Viparita Dandasana)",
            "Upward Plank Pose (Purvottanasana)",
            "Virasana (Vajrasana)",
            "Warrior III Pose (Virabhadrasana III)",
            "Warrior II Pose (Virabhadrasana II)",
            "Warrior I Pose (Virabhadrasana I)",
            "Wide-Angle Seated Forward Bend Pose (Upavistha Konasana)",
            "Wide-Legged Forward Bend Pose (Prasarita Padottanasana)",
            "Wild Thing Pose (Camatkarasana)",
            "Wind Relieving Pose (Pawanmuktasana)",
            "Yogic Sleep Pose",
            "Reverse Warrior Pose (Viparita Virabhadrasana)"
        ]

    def process_keypoints_rectangle(self, keypoints):
        min_x = keypoints[1]
        min_y = keypoints[0]
        max_x = keypoints[1]
        max_y = keypoints[0]
        for i in range(0, 17):
            point_x = keypoints[i * 2 + 1]
            point_y = keypoints[i * 2 + 0]
            if point_x > max_x:
                max_x = point_x
            if point_x < min_x:
                min_x = point_x
            if point_y > max_y:
                max_y = point_y
            if point_y < min_y:
                min_y = point_y
        for i in range(0, 17):
            keypoints[i * 2 + 1] = keypoints[i * 2 + 1] - min_x
            keypoints[i * 2 + 0] = keypoints[i * 2 + 0] - min_y

        return keypoints

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

    def read_yoga_data_from_file(self, file_name, with_confidence_threshold):
        x = []
        y = []
        with open(file_name, newline='\n') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                if int(float(row[-1])) in range(0, len(self.label_names_82)):
                    x_temp = []
                    percents = []
                    for i in range(0, 17):
                        x_temp.append(float(row[1 + i * 3]))
                        x_temp.append(float(row[1 + i * 3 + 1]))
                        percents.append(float(row[1 + i * 3 + 2]))
                    if (with_confidence_threshold and (min(percents) >= self.confidence_threshold)) or (
                    not with_confidence_threshold):
                        if self.keypoint_processing_id == 1:
                            x_temp = self.process_keypoints_rectangle(x_temp)
                        if self.keypoint_processing_id == 2:
                            x_temp = self.process_keypoints_normalized_square(x_temp)
                        x.append(x_temp)
                        y.append(int(float(row[-1])))
        return np.array(x), np.array(y)

    def read_data(self, keypoint_processing_id):
        self.keypoint_processing_id = keypoint_processing_id
        self.x, self.y = self.read_yoga_data_from_file(self.train_data_file, with_confidence_threshold=True)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x, self.y, test_size=0.1)
        self.x_test, self.y_test = self.read_yoga_data_from_file(self.test_data_file, with_confidence_threshold=False)
        self.x_strict_test, self.y_strict_test = self.read_yoga_data_from_file(self.test_data_file,
                                                                               with_confidence_threshold=True)

    def make_report(self, y_true, y_predicted, prefix_name=""):
        classes = [i for i in range(0, len(self.label_names_82))]

        bas = balanced_accuracy_score(y_true, y_predicted)
        mlflow.log_metric(prefix_name + 'balanced_accuracy_all', bas)
        print('balanced accuracy score', bas)
        acs = accuracy_score(y_true, y_predicted)
        mlflow.log_metric(prefix_name + 'accuracy_all', acs)
        print('accuracy score', acs)
        # http://gabrielelanaro.github.io/blog/2016/02/03/multiclass-evaluation-measures.html

        cm = confusion_matrix(y_true, y_predicted, labels=classes)
        fig = plt.figure(figsize=(150, 150))
        sb.set(font_scale=2.2)
        sb.heatmap(cm, annot=True, annot_kws={"size": 60})
        cm_name = prefix_name + "confusion_matrix.jpg"
        plt.savefig(cm_name)
        mlflow.log_artifact(cm_name, 'Data metrics')

        number_of_data = {}
        for cl in classes:
            number_of_data[cl] = 0

        for result in y_true:
            number_of_data[result] += 1

        fig = plt.figure(figsize=(50, 50))
        sb.set(font_scale=1.2)
        d = [[i, number_of_data[i]] for i in number_of_data.keys()]
        color = plt.cm.copper_r(list(number_of_data.values()))
        sb.barplot(d, errorbar=(None), palette=color)
        nd_name = prefix_name + "number_of_data.jpg"
        plt.savefig(nd_name)
        mlflow.log_artifact(nd_name, 'Data metrics')

        class_metrics = []

        micro_precision_numerator = 0
        micro_precision_denominator = 0
        macro_precision_sum = 0
        micro_recall_numerator = 0
        micro_recall_denominator = 0
        macro_recall_sum = 0
        micro_specificity_numerator = 0
        micro_specificity_denominator = 0
        macro_specificity_sum = 0

        for cl in range(0, len(cm)):
            tp = cm[cl][cl]
            fp = sum(cm[:, cl]) - cm[cl][cl]
            fn = sum(cm[cl, :]) - cm[cl][cl]
            tn = sum(sum(cm)) - tp - fp - fn

            accuracy = round((tp + tn) / (tp + fp + tn + fn), 3)
            mlflow.log_metric(prefix_name + 'accuracy_' + str(cl), accuracy)

            precision = round(tp / (tp + fp), 3)
            mlflow.log_metric(prefix_name + 'precision_' + str(cl), precision)
            micro_precision_numerator += tp
            micro_precision_denominator += tp + fp
            if precision >=0 :
                macro_precision_sum += precision

            recall = round(tp / (tp + fn), 3)
            mlflow.log_metric(prefix_name + 'recall_' + str(cl), recall)
            micro_recall_numerator += tp
            micro_recall_denominator += tp + fn
            if recall >=0:
                macro_recall_sum += recall

            specificity = round(tn / (tn + fp), 3)
            mlflow.log_metric(prefix_name + 'specificity_' + str(cl), specificity)
            micro_specificity_numerator += tn
            micro_specificity_denominator += tn + fp
            if specificity >=0:
                macro_specificity_sum += specificity

            f1_score = round(2 * (precision * recall) / (precision + recall), 3)
            mlflow.log_metric(prefix_name + 'f1_score_' + str(cl), f1_score)

            class_metrics.append([accuracy, precision, recall, specificity, f1_score])

        micro_precision_all = round(micro_precision_numerator / micro_precision_denominator, 3)
        mlflow.log_metric(prefix_name + "micro_precision_all", micro_precision_all)
        macro_precision_all = round(macro_precision_sum / len(cm), 3)
        mlflow.log_metric(prefix_name + "macro_precision_all", macro_precision_all)
        micro_recall_all = round(micro_recall_numerator / micro_recall_denominator, 3)
        mlflow.log_metric(prefix_name + "micro_recall_all", micro_recall_all)
        macro_recall_all = round(macro_recall_sum / len(cm), 3)
        mlflow.log_metric(prefix_name + "macro_recall_all", macro_recall_all)
        micro_specificity_all = round(micro_specificity_numerator / micro_specificity_denominator, 3)
        mlflow.log_metric(prefix_name + "micro_specificity_all", micro_specificity_all)
        macro_specificity_all = round(macro_specificity_sum / len(cm), 3)
        mlflow.log_metric(prefix_name + "macro_specificity_all", macro_specificity_all)
        micro_f1_score_all = round(2 * (micro_recall_all * micro_precision_all) / (micro_recall_all + micro_precision_all),3)
        mlflow.log_metric(prefix_name + "micro_f1_score_all", micro_f1_score_all)
        macro_f1_score_all = round(2 * (macro_recall_all * macro_precision_all) / (macro_recall_all + macro_precision_all),3)
        mlflow.log_metric(prefix_name + "macro_f1_score_all", macro_f1_score_all)

        clm_df = pd.DataFrame(class_metrics, columns=['accuracy', 'precision', 'recall', 'specificity', 'f1_score'])
        fig = plt.figure(figsize=(100, 100))
        sb.set(font_scale=2.2)
        sb.heatmap(clm_df, annot=True, annot_kws={"size": 40},
                   xticklabels=['accuracy', 'precision', 'recall', 'specificity', 'f1_score'],
                   yticklabels=classes)
        clm_name = prefix_name + "class_metrics.jpg"
        plt.savefig(clm_name)
        mlflow.log_artifact(clm_name, 'Data metrics')

        cm_n = confusion_matrix(y_true, y_predicted, labels=classes, normalize='true')
        fig = plt.figure(figsize=(150, 150))
        sb.set(font_scale=0.2)
        sb.heatmap(cm_n, annot=True, annot_kws={"size": 40})
        cm_n_name = prefix_name + "confusion_matrix_normalized.jpg"
        plt.savefig(cm_n_name)
        mlflow.log_artifact(cm_n_name, 'Data metrics')

        plt.close("all")

    def linear_function(self, model_name, learning_rate=0.001, batch_size=16, epoch=500):
        mlflow.tensorflow.autolog(registered_model_name=model_name)
        mlflow.log_param('train data file', self.train_data_file)
        mlflow.log_param('test data file', self.test_data_file)
        mlflow.log_param('keypoint confidence threshold', self.confidence_threshold)

        linear = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(82, activation='softmax')
        ])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.001,
                patience=15,
                verbose=1,
            )
        ]

        linear.build(input_shape=(None, 34))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        linear.compile(optimizer=optimizer,
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                       metrics=['accuracy'])

        fit_time_start = time.time()
        linear.fit(self.x_train, self.y_train, epochs=epoch, batch_size=batch_size, callbacks=callbacks,
                   validation_data=(self.x_val, self.y_val))
        fit_time_end = time.time()
        mlflow.log_metric('fit_time', fit_time_end - fit_time_start)

        test_time_start = time.time()
        y_predicted_all = linear.predict(self.x_test)
        y_predicted = np.argmax(y_predicted_all, 1)
        test_time_end = time.time()
        mlflow.log_metric('test_time', test_time_end - test_time_start)
        self.make_report(self.y_test, y_predicted, "test_")

        test_time_start = time.time()
        y_predicted_all_strict = linear.predict(self.x_strict_test)
        y_predicted_strict = np.argmax(y_predicted_all_strict, 1)
        test_time_end = time.time()
        mlflow.log_metric('strict_test_time', test_time_end - test_time_start)
        self.make_report(self.y_strict_test, y_predicted_strict, "test_strict_")

    def knn_function(self, model_name, k_value):
        mlflow.sklearn.autolog(registered_model_name=model_name)
        mlflow.log_param('train data file', self.train_data_file)
        mlflow.log_param('test data file', self.test_data_file)
        mlflow.log_param('keypoint confidence threshold', self.confidence_threshold)

        knn = KNeighborsClassifier(n_neighbors=k_value)

        fit_time_start = time.time()
        knn.fit(self.x, self.y)
        fit_time_end = time.time()
        mlflow.log_metric('fit_time', fit_time_end - fit_time_start)

        test_time_start = time.time()
        y_predicted = knn.predict(self.x_test)
        test_time_end = time.time()
        mlflow.log_metric('test_time', test_time_end - test_time_start)
        self.make_report(self.y_test, y_predicted, "test_")

        strict_test_time_start = time.time()
        y_strict_predicted = knn.predict(self.x_strict_test)
        strict_test_time_end = time.time()
        mlflow.log_metric('strict_test_time', strict_test_time_end - strict_test_time_start)
        self.make_report(self.y_strict_test, y_strict_predicted, "test_strict_")

    def rf_function(self, model_name, max_depth):
        mlflow.sklearn.autolog(registered_model_name=model_name)
        mlflow.log_param('train data file', self.train_data_file)
        mlflow.log_param('test data file', self.test_data_file)
        mlflow.log_param('keypoint confidence threshold', self.confidence_threshold)

        rf = RandomForestClassifier(max_depth=max_depth)

        fit_time_start = time.time()
        rf.fit(self.x, self.y)
        fit_time_end = time.time()
        mlflow.log_metric('fit_time', fit_time_end - fit_time_start)

        test_time_start = time.time()
        y_predicted = rf.predict(self.x_test)
        test_time_end = time.time()
        mlflow.log_metric('test_time', test_time_end - test_time_start)
        self.make_report(self.y_test, y_predicted, "test_")

        strict_test_time_start = time.time()
        y_strict_predicted = rf.predict(self.x_strict_test)
        strict_test_time_end = time.time()
        mlflow.log_metric('strict_test_time', strict_test_time_end - strict_test_time_start)
        self.make_report(self.y_strict_test, y_strict_predicted, "test_strict_")

def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    logging.basicConfig(level=logging.WARN)
    #logger = logging.getLogger(__name__)

    data_dictionary = {
        'yoga_thunder_train.csv': 'yoga_thunder_test.csv',
        'yoga_lightning_train.csv': 'yoga_lightning_test.csv'
    }

    # experiments:
    # 1. normalized, without keypoint processing
    # 2. normalized, with keypoint processing (rectangle)
    # 3. normalized, with keypoint processing (normalized square, hip middle)
    exp_name_list = ["Yoga poses 82: without keypoint processing",
                     "Yoga poses 82: with keypoint processing (rectangle)",
                     "Yoga poses 82: with keypoint processing (normalized square)"]
    kp_list = ["NKP", "KPR", "KPNS"]
    dd_id = {'yoga_thunder_train.csv': "T",
             'yoga_lightning_train.csv': "L"}

    for i in range(0, 3):
        exp_name = exp_name_list[i]
        kp_name = kp_list[i]
        keypoint_processing_id = i
        try:
            mlflow.create_experiment(exp_name)
        except:
            print('Experiment already exists:', exp_name)
        id = mlflow.get_experiment_by_name(exp_name).experiment_id

        print('Starting training for experiment:', exp_name)
        for d_l in data_dictionary:
            dd_name = dd_id[d_l]
            train_data_file = d_l
            test_data_file = data_dictionary[d_l]
            print('Current train and test files:', train_data_file, test_data_file)

            trainer = yoga_pose_classification_trainer(train_data_file, test_data_file, 0.3)
            trainer.read_data(keypoint_processing_id)

            epoch = 500
            learning_rate = 0.001
            batch_size = 16
            with mlflow.start_run(run_name='Linear Neural Network', experiment_id=id):
                model_name = "NN" + "_" + dd_name + "_" + kp_name
                print("NN FUNCTION", model_name)
                trainer.linear_function(model_name, learning_rate, batch_size, epoch)

            batch_size = 8
            with mlflow.start_run(run_name='Linear Neural Network', experiment_id=id):
                model_name = "NN" + "_" + dd_name + "_" + kp_name
                print("NN FUNCTION", model_name)
                trainer.linear_function(model_name, learning_rate, batch_size, epoch)

            learning_rate = 0.0001
            batch_size = 16
            with mlflow.start_run(run_name='Linear Neural Network', experiment_id=id):
                model_name = "NN" + "_" + dd_name + "_" + kp_name
                print("NN FUNCTION", model_name)
                trainer.linear_function(model_name, learning_rate, batch_size, epoch)

            batch_size = 8
            with mlflow.start_run(run_name='Linear Neural Network', experiment_id=id):
                model_name = "NN" + "_" + dd_name + "_" + kp_name
                print("NN FUNCTION", model_name)
                trainer.linear_function(model_name, learning_rate, batch_size, epoch)

            with mlflow.start_run(run_name="K-Nearest Neighbor", experiment_id=id):
                model_name = "KNN" + "_" + dd_name + "_" + kp_name
                print("KNN FUNCTION", model_name)
                trainer.knn_function(model_name, 5)

            with mlflow.start_run(run_name="K-Nearest Neighbor", experiment_id=id):
                model_name = "KNN" + "_" + dd_name + "_" + kp_name
                print("KNN FUNCTION", model_name)
                trainer.knn_function(model_name, 10)

            with mlflow.start_run(run_name="K-Nearest Neighbor", experiment_id=id):
                model_name = "KNN" + "_" + dd_name + "_" + kp_name
                print("KNN FUNCTION", model_name)
                trainer.knn_function(model_name, 20)

            with mlflow.start_run(run_name='Random Forest', experiment_id=id):
                model_name = "RF" + "_" + dd_name + "_" + kp_name
                print("RF FUNCTION", model_name)
                trainer.rf_function(model_name, 10)

            with mlflow.start_run(run_name='Random Forest', experiment_id=id):
                model_name = "RF" + "_" + dd_name + "_" + kp_name
                print("RF FUNCTION", model_name)
                trainer.rf_function(model_name, 25)

            with mlflow.start_run(run_name='Random Forest', experiment_id=id):
                model_name = "RF" + "_" + dd_name + "_" + kp_name
                print("RF FUNCTION", model_name)
                trainer.rf_function(model_name, None)


if __name__ == "__main__":
    main()
