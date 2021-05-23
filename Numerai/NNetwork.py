import keras as K
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras import models
from matplotlib import pyplot as plt
from ModelBase import ModelBase

from defines import *

#tensorboard --logdir logs/fit

THIS_MODEL_PATH = MODEL_PATH + '/nnModels/nn'

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


class NNModel(ModelBase):
    lr = 0.001
    epochs = 20000
    batchSize = 8192

    stopping = K.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=100,
        mode="auto",
        baseline=None,
        min_delta=0.001,
        restore_best_weights=True)


    checkpoint = K.callbacks.ModelCheckpoint(filepath=THIS_MODEL_PATH+'/tmp', 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')

    def __init__(self, name=None):
        if name:
            self.model = models.load_model(THIS_MODEL_PATH + name)
            return

        neurons = 196
        inp = layers.Input((316,))
        
        out = layers.Dense(neurons)(inp)
        out = layers.ReLU()(out)
        out = layers.BatchNormalization()(out)
        for i in range(6):
            out = self.addResBlock(out, neurons, 0.4)

        out = layers.Dense(neurons, 'relu')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(1)(out)
        out = K.activations.sigmoid(out)

        self.model = K.Model(inputs = inp, outputs = out)

        self.model.compile(optimizer=K.optimizers.Adam(),
              #loss=K.losses.mean_squared_error,
              loss=K.losses.binary_crossentropy,
              metrics=[])

        self.model.summary()



    def addResBlock(self, inp, sz, d):
        m = layers.Dense(sz)(inp)
        mt = layers.ReLU()(m)
        mt = layers.BatchNormalization()(mt)
        mt = layers.Dropout(d)(mt)

        mt = layers.Dense(sz)(mt)
        mt = layers.ReLU()(m)
        mt = layers.BatchNormalization()(mt)
        mt = layers.Dropout(d)(mt)
        return layers.Add()([inp, mt])

    #def addDenseBlock(self, sz):
    #    self.model.add(layers.Dense(sz))
    #    self.model.add(layers.ReLU())
    #    self.model.add(layers.BatchNormalization())
    #    self.model.add(layers.Dropout(0.15))

    def fit(self, features, targets, valFeatures, valTargets):
        #features['era'] = features['era'].apply(lambda x: (float(x[3:]) / features.shape[0]))
        #valFeatures['era'] = valFeatures['era'].apply(lambda x: (float(x[3:]) / valFeatures.shape[0]))

        hist = self.model.fit(x=features.values, y=targets.values, epochs=self.epochs, 
                        batch_size=self.batchSize, #steps_per_epoch=10,
                        validation_data=(valFeatures.values, valTargets.values),
                        shuffle=True, callbacks=[self.stopping, tensorboard])

        self.model.save(THIS_MODEL_PATH + '-{}'.format(
            round(hist.history['val_loss'][-1], 3)))

        NNModel.plotLoss('training', hist)

    def predict(self, features, savePath=None):
        p = self.model.predict(x=features.values)
        if not savePath:
            return p
        #p = pd.DataFrame(p, index=features.index, columns='feature_nnpred')
        #p.to_hdf(savePath + "data.h5", key='nnout')

