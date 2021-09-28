import datetime
import numpy as np
import tensorflow as tf


FAST_MODE = True # Ignore reporting on scores we don't need (like when 

DATA_TYPE = np.float32

TARGET_NAME = 'target'
PREDICTION_NAME = 'prediction'

DIR = "./data/"
MODEL_PATH = './models'  

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

PAYOUT_MULTIPLIER = 0.55
np.random.seed(1)
