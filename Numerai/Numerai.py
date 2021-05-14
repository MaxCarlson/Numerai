# Ideas
# Maybe add difference in eras as feature?

# Feature Engineering
# # # # # # # # # # # # 
# Use autoencoder for feature engineering!
# Try count-encoding, switch feature value to it's respective frequency of occurance in that feature
# Some sort of feature selection search?
# Take only the most highly correlated features? Univariate Feature Selection?
# Mean/Median/etc of feature catagories

import numerapi
import numpy as np
import pandas as pd
from xgboost import XGBRegressor 
import csv
import keras
import tensorflow as tf
from keras import layers
from keras import models
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

NAPI = numerapi.NumerAPI(verbosity="info")


DIR = "./data/"
TARGET_NAME = 'target'
PREDICTION_NAME = 'prediction'

# Download new data
NAPI.download_current_dataset(dest_path=DIR, unzip=True)


# Submissions are scored by spearman correlation
def correlation(predictions, targets):
    ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]

# convenience method for scoring
def score(df):
    return correlation(df[PREDICTION_NAME], df[TARGET_NAME])

# Payout is just the score cliped at +/-25%
def payout(scores):
    return scores.clip(lower=-0.25, upper=0.25)



class AutoEncoder():
    
    lr = 0.009
    epochs = 100
    batchSize = 128
    stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="auto",
        baseline=None,
        restore_best_weights=True)

    decay = keras.optimizers.schedules.ExponentialDecay(
        lr, decay_steps=3400*2, decay_rate=0.95)

    featureThreshold = 0.01

    def __init__(self):
        self.encoder = None

        # Note: Current best:
        # 310, 256, 196, 128: lr=0.009, decay_steps=6800, decay_rate=0.95
        # act=tanh, out=sigmoid, loss=binary_entropy, optimizer=adam
        # Epochs=3, val_loss=0.606
        # Features > 0.01: 0.0148, 0.0131

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
        self.model.add(layers.Dense(128))
        self.model.add(layers.LeakyReLU(alpha=0.3))
        self.model.add(layers.BatchNormalization())

        self.model.add(layers.Dense(128, name='OutputLayer'))
        self.model.add(layers.LeakyReLU(alpha=0.3))
        self.model.add(layers.BatchNormalization())
        
        self.model.add(layers.Dense(128))
        self.model.add(layers.LeakyReLU(alpha=0.3))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(196))
        self.model.add(layers.LeakyReLU(alpha=0.3))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(256))
        self.model.add(layers.LeakyReLU(alpha=0.3))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(310, activation='sigmoid'))
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.decay), 
              #loss=keras.losses.mean_squared_error,
              loss=keras.losses.binary_crossentropy,
              metrics=[])

    def fit(self, features, targets, val):
        if False:
            self.model = keras.models.load_model('./aeModels/autoencoder-0.423')

        else:
            hist = self.model.fit(x=fetures.values, y=targets.values, epochs=self.epochs, 
                                     batch_size=self.batchSize, #steps_per_epoch=10,
                                     validation_data=(val.values, val.values),
                                     shuffle=True, callbacks=[self.stopping])

            #self.model.save('./aeModels/autoencoder-{}'.format(round(hist.history['val_loss'][-1], 3)))
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

# Read the csv file into a pandas Dataframe as float16 to save space
def read_csv(file_path):
    with open(file_path, 'r') as f:
        column_names = next(csv.reader(f))

    dtypes = {x: np.float16 for x in column_names if x.startswith(('feature', TARGET_NAME))}
    df = pd.read_csv(file_path, dtype=dtypes, index_col=0)
    return df


def main():
    ae = AutoEncoder()
    print("Loading data...")
    # The training data is used to train your model how to predict the targets.
    training_data = read_csv("data/numerai_dataset_263/numerai_training_data.csv")
    # The tournament data is the data that Numerai uses to evaluate your model.
    tournament_data = read_csv("data/numerai_dataset_263/numerai_tournament_data.csv")
    validation_data = tournament_data[tournament_data.data_type == "validation"]
    #printCorrelation(training_data)

    feature_names = [
        f for f in training_data.columns if f.startswith("feature")
    ]
    print(f"Loaded {len(feature_names)} features")

    #print('Printing features: \n', validation_data[feature_names])
    #print('Printing targets: \n', validation_data['target'])

    ae.fit(training_data[feature_names], training_data[TARGET_NAME], validation_data[feature_names])
    aeoutTrain = ae.encode(training_data[feature_names])
    aeoutVal = ae.encode(validation_data[feature_names])


    train_corr_matrix = AutoEncoder.printCorrelation(aeoutTrain, training_data[TARGET_NAME])
    valid_corr_matrix = AutoEncoder.printCorrelation(aeoutVal, validation_data[TARGET_NAME])


    return


    #training_data[PREDICTION_NAME] = model.predict(training_data[feature_names], training_data[TARGET_NAME])
    #tournament_data[PREDICTION_NAME] = model.predict(tournament_data[feature_names], tournament_data[TARGET_NAME])

    ## Check the per-era correlations on the training set (in sample)
    #train_correlations = training_data.groupby("era").apply(score)
    #print(f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std(ddof=0)}")
    #print(f"On training the average per-era payout is {payout(train_correlations).mean()}")
    #
    #"""Validation Metrics"""
    ## Check the per-era correlations on the validation set (out of sample)
    #validation_data[PREDICTION_NAME] = tournament_data[PREDICTION_NAME]
    #validation_correlations = validation_data.groupby("era").apply(score)
    #print(f"On validation the correlation has mean {validation_correlations.mean()} and "
    #      f"std {validation_correlations.std(ddof=0)}")
    #print(f"On validation the average per-era payout is {payout(validation_correlations).mean()}")
    #
    ## Check the "sharpe" ratio on the validation set
    #validation_sharpe = validation_correlations.mean() / validation_correlations.std(ddof=0)
    #print(f"Validation Sharpe: {validation_sharpe}")
    



if __name__ == "__main__":
    main()
