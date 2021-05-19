import datetime
import tensorflow as tf

TARGET_NAME = 'target'
PREDICTION_NAME = 'prediction'

DIR = "./data/"
MODEL_PATH = './models'
CURRENT_DATASET = "numerai_dataset_264/"
DATASET_PATH = DIR + CURRENT_DATASET

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
