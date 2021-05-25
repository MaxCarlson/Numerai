import keras as K
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras import models
from matplotlib import pyplot as plt
from ModelBase import ModelBase

import torch
from pytorch_tabnet.tab_model import TabNetRegressor
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
    lr = 1e-4
    epochs = 20000
    batchSize = 512

    stopping = K.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="auto",
        baseline=None,
        min_delta=0.0002,
        restore_best_weights=True)


    checkpoint = K.callbacks.ModelCheckpoint(filepath=THIS_MODEL_PATH+'/tmp', 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')

    reduce = K.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=10,
        verbose=0,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=0,
    )

    def __init__(self, name=None):
        if name:
            self.model = models.load_model(THIS_MODEL_PATH + name)
            return

        #neurons = 196
        #inp = layers.Input((310,))
        #
        #out = layers.Dense(neurons)(inp)
        #out = layers.ReLU()(out)
        #out = layers.BatchNormalization()(out)
        #out = layers.Dense(neurons)(out)
        #out = layers.ReLU()(out)
        #out = layers.BatchNormalization()(out)
        #out = layers.Dense(neurons)(out)
        #out = layers.ReLU()(out)
        #out = layers.BatchNormalization()(out)
        #
        #out = layers.Dense(neurons, 'relu')(out)
        #out = layers.BatchNormalization()(out)
        #out = layers.Dense(1)(out)
        #out = K.activations.sigmoid(out)
        #
        #self.model = K.Model(inputs = inp, outputs = out)
        #
        #self.model.compile(optimizer=K.optimizers.Adam(),
        #      loss=K.losses.mean_squared_error,
        #      #loss=K.losses.binary_crossentropy,
        #      metrics=[])
        
        #lr_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(torch.optim.Adam,  
        lr_sch = torch.optim.lr_scheduler.ExponentialLR

        self.model = TabNetRegressor(device_name='cuda', n_d=8, n_a=8, 
                                     n_steps=5, lambda_sparse=0.0001, gamma=1.5,
                                     optimizer_params=dict(lr=0.004), 
                                     scheduler_fn=lr_sch, scheduler_params={'gamma': 0.94})#, 'verbose': True})
        torch.cuda.set_device(0)
        return
        #for i in range(1):
        #    out = self.addResBlock(out, neurons, 0.1)
        #
        #
        #self.model.summary()


    def addResBlock(self, inp, sz, d):
        m = layers.Dense(sz)(inp)
        mt = layers.ReLU()(m)
        mt = layers.BatchNormalization()(mt)
        mt = layers.Dropout(d)(mt)

        mt = layers.Dense(sz)(mt)
        mt = layers.ReLU()(m)
        mt = layers.BatchNormalization()(mt)
        mt = layers.Dropout(d)(mt)
        return mt #layers.Add()([inp, mt])

    #def addDenseBlock(self, sz):
    #    self.model.add(layers.Dense(sz))
    #    self.model.add(layers.ReLU())
    #    self.model.add(layers.BatchNormalization())
    #    self.model.add(layers.Dropout(0.15))

    def fit(self, features, targets, valFeatures, valTargets):
        #features['era'] = features['era'].apply(lambda x: (float(x[3:]) / features.shape[0]))
        #valFeatures['era'] = valFeatures['era'].apply(lambda x: (float(x[3:]) / valFeatures.shape[0]))

        self.model.device_name ='cuda'
        self.model.fit(features.values, targets.values.reshape(-1,1), eval_set=[
            (features.values, targets.values.reshape(-1,1)),
            (valFeatures.values, valTargets.values.reshape(-1,1))],
                       eval_name=['train', 'valid'],
                       eval_metric=['mse'],#'rmse', 'mae', 
                       batch_size=2048, virtual_batch_size=256, num_workers=0)
        self.model.save_model('tabnet.mdl')

        #hist = self.model.fit(x=features.values, y=targets.values, epochs=self.epochs, 
        #                batch_size=self.batchSize, #steps_per_epoch=10,
        #                validation_data=(valFeatures.values, valTargets.values),
        #                shuffle=True, callbacks=[self.stopping, tensorboard, self.reduce])
        #
        #self.model.save(THIS_MODEL_PATH + '-{}'.format(
        #    round(np.min(hist.history['val_loss']), 3)))
        #
        #NNModel.plotLoss('training', hist)

    def predict(self, features, savePath=None):
        self.model.load_model('tabnet.mdl.zip')
        p = self.model.predict(features.values)
        #p = self.model.predict(x=features.values)
        return p

