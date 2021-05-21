import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras import layers
from keras import models
from ModelBase import ModelBase

from defines import *

AE_MODELS = MODEL_PATH + '/aeModels/'
THIS_MODEL_PATH = AE_MODELS + 'autoencoder-'

class AutoEncoder(ModelBase):
    
    lr = 0.008
    epochs = 1500
    batchSize = 8192

    stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=200,
        mode="auto",
        baseline=None,
        min_delta=0.0005,
        restore_best_weights=True)

    #decay = keras.optimizers.schedules.ExponentialDecay(
    #    lr, decay_steps=3400, decay_rate=0.95)

    def __init__(self):
        self.encoder = None

        self.model = models.Sequential()
        self.model.add(layers.Dense(310))
        self.model.add(layers.LeakyReLU(alpha=0.01))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(256))
        self.model.add(layers.LeakyReLU(alpha=0.01))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(196))
        self.model.add(layers.LeakyReLU(alpha=0.01))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(162))
        self.model.add(layers.LeakyReLU(alpha=0.01))
        self.model.add(layers.BatchNormalization())
        #self.model.add(layers.Dense(146))
        #self.model.add(layers.LeakyReLU(alpha=0.01))
        #self.model.add(layers.BatchNormalization())

        self.model.add(layers.Dense(128, name='OutputLayer'))
        self.model.add(layers.LeakyReLU(alpha=0.01))
        self.model.add(layers.BatchNormalization())
        
        #self.model.add(layers.Dense(146))
        #self.model.add(layers.LeakyReLU(alpha=0.01))
        #self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(162))
        self.model.add(layers.LeakyReLU(alpha=0.01))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(196))
        self.model.add(layers.LeakyReLU(alpha=0.01))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(256))
        self.model.add(layers.LeakyReLU(alpha=0.01))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(310, activation='sigmoid'))
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), 
              #loss=keras.losses.mean_squared_error,
              loss=keras.losses.binary_crossentropy,
              metrics=[])

        #self.model.summary()

    def load(self, name):
        self.model = keras.models.load_model(THIS_MODEL_PATH + name)
        aeOutput = self.model.get_layer(name='OutputLayer').output
        self.encoder = keras.Model(inputs=self.model.input, outputs=aeOutput)

    def fit(self, features, val):
        if False:
            self.model = keras.models.load_model(THIS_MODEL_PATH + '0.423')

        else:
            hist = self.model.fit(x=features.values, y=features.values, epochs=self.epochs, 
                                     batch_size=self.batchSize, #steps_per_epoch=10,
                                     validation_data=(val.values, val.values),
                                     shuffle=True, callbacks=[self.stopping, tensorboard])

            self.model.save(THIS_MODEL_PATH + '{}'.format(round(hist.history['val_loss'][-1], 3)))
        aeOutput = self.model.get_layer(name='OutputLayer').output
        self.encoder = keras.Model(inputs=self.model.input, outputs=aeOutput)

    def encode(self, features):
        p = self.encoder.predict(x=features.values)
        return p

    def saveData(self, training_data, tournament_data, feature_names):
        print('Generating encoded data...')

        mcs = ['era', 'data_type', 'target']
        new_cols = mcs + ['feature{}'.format(f) for f in range(aeoutTrain.shape[1])]
        def pands(data, name):
            out      = self.encode(data[feature_names])
            saved = pd.DataFrame(data=out, columns=new_cols)
            saved[mcs] = data[mcs]
            saved.to_csv(AE_MODELS + name)

        pands(training_data, 'numerai_training_data.csv')
        #pands(tournament_data, 'numerai_tournament_data.csv')

        print('Encoded data saved...')


