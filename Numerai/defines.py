import datetime
import numpy as np
import tensorflow as tf

DATA_TYPE = np.float32

TARGET_NAME = 'target'
PREDICTION_NAME = 'prediction'

DIR = "./data/"
MODEL_PATH = './models'
CURRENT_DATASET = "numerai_dataset_265/"
DATASET_PATH = DIR + CURRENT_DATASET

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
