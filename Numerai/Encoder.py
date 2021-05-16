import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras import layers
from keras import models

from defines import *

THIS_MODEL_PATH = 'aeModels/autoencoder'

class AutoEncoder():
    
    lr = 0.008
    epochs = 100
    batchSize = 128

    stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="auto",
        baseline=None,
        min_delta=0.005,
        restore_best_weights=True)

    decay = keras.optimizers.schedules.ExponentialDecay(
        lr, decay_steps=3400, decay_rate=0.95)

    featureThreshold = 0.01

    def __init__(self):
        self.encoder = None

        # Note: Current best:

        self.model = models.Sequential()
        self.model.add(layers.Dense(310))
        self.model.add(layers.LeakyReLU(alpha=0.3))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(256))
        self.model.add(layers.LeakyReLU(alpha=0.3))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(196))
        self.model.add(layers.LeakyReLU(alpha=0.3))
        self.model.add(layers.BatchNormalization())
        #self.model.add(layers.Dense(128))
        #self.model.add(layers.LeakyReLU(alpha=0.3))
        #self.model.add(layers.BatchNormalization())

        self.model.add(layers.Dense(128, name='OutputLayer'))
        self.model.add(layers.LeakyReLU(alpha=0.3))
        self.model.add(layers.BatchNormalization())
        
        #self.model.add(layers.Dense(128))
        #self.model.add(layers.LeakyReLU(alpha=0.3))
        #self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(196))
        self.model.add(layers.LeakyReLU(alpha=0.3))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(256))
        self.model.add(layers.LeakyReLU(alpha=0.3))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(310, activation='sigmoid'))
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), 
              #loss=keras.losses.mean_squared_error,
              loss=keras.losses.binary_crossentropy,
              metrics=[])

        self.model.summary()


    def fit(self, features, val):
        if False:
            self.model = keras.models.load_model(MODEL_PATH + THIS_MODEL_PATH + '-0.423')

        else:
            hist = self.model.fit(x=features.values, y=features.values, epochs=self.epochs, 
                                     batch_size=self.batchSize, #steps_per_epoch=10,
                                     validation_data=(val.values, val.values),
                                     shuffle=True, callbacks=[self.stopping])

            self.model.save(MODEL_PATH + THIS_MODEL_PATH + '-{}'.format(round(hist.history['val_loss'][-1], 3)))
        aeOutput = self.model.get_layer(name='OutputLayer').output
        self.encoder = keras.Model(inputs=self.model.input, outputs=aeOutput)

    def encode(self, features):
        p = self.encoder.predict(x=features.values)
        return p

    @staticmethod
    def printCorrelation(aeout, targets):
        aeout = pd.DataFrame(aeout)
        aeout[TARGET_NAME] = targets.values

        corr_matrix = aeout.corr()
        corr_matrix = corr_matrix[TARGET_NAME].sort_values(ascending=False)
        print(corr_matrix)
        return corr_matrix
