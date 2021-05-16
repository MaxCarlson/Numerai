import keras as K
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras import models

from defines import *

THIS_MODEL_PATH='/nnModels/nn'

# TF differentiable correaltion function(s)?
def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)

def correlation_tf(x, y):    
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x-mx, y-my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym))        
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return r_num / r_den


class NNModel():
    lr = 0.005
    epochs = 100
    batchSize = 128

    stopping = K.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="auto",
        baseline=None,
        min_delta=0.005,
        restore_best_weights=True)

    def __init__(self, name=None):
        if name:
            self.model = K.models.load_model(MODEL_PATH + THIS_MODEL_PATH + name)
            return

        self.model = models.Sequential()
        self.model.add(K.Input(shape=(310,)))
        for i in range(20):
            self.addDenseBlock(128)

        self.model.add(layers.Dense(1))
        self.model.add(layers.Activation(K.activations.sigmoid))

        self.model.compile(optimizer=K.optimizers.Adam(learning_rate=self.lr), 
              #loss=K.losses.mean_squared_error,
              loss=K.losses.binary_crossentropy,
              metrics=[])

        #self.model.summary()

    def addDenseBlock(self, sz):
        self.model.add(layers.Dense(sz))
        self.model.add(layers.LeakyReLU(alpha=0.3))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.15))


    def fit(self, features, targets, valFeatures, valTargets):
        hist = self.model.fit(x=features.values, y=targets.values, epochs=self.epochs, 
                        batch_size=self.batchSize, #steps_per_epoch=10,
                        validation_data=(valFeatures.values, valTargets.values),
                        shuffle=True, callbacks=[self.stopping])

        self.model.save(MODEL_PATH + THIS_MODEL_PATH + '-{}'.format(round(hist.history['val_loss'][-1], 3)))

    def predict(self, features):
        p = self.model.predict(x=features.values)
        return p