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
TARGET_NAME = f"target"
PREDICTION_NAME = f"prediction"

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

def printCorrelation(df):
    corr_matrix = df.corr()
    corr_matrix = corr_matrix['target'].sort_values(ascending=False)
    print(corr_matrix)
    return corr_matrix

class AutoEncoder():
    
    lr = 0.009
    epochs = 100
    batchSize = 128
    stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
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
        #self.model.add(layers.Dense(128))
        #self.model.add(layers.LeakyReLU(alpha=0.3))
        #self.model.add(layers.BatchNormalization())
        #self.model.add(layers.Dense(64))
        self.model.add(layers.Dense(128, name='OutputLayer'))
        self.model.add(layers.LeakyReLU(alpha=0.3))
        #self.model.add(layers.BatchNormalization()) ?
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
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.decay), 
              #loss=keras.losses.mean_squared_error,
              loss=keras.losses.binary_crossentropy,
              metrics=[])

    def fit(self, data, val):
        if False:
            self.model = keras.models.load_model('./aeModels/autoencoder-0.423')

        else:
            hist = self.model.fit(x=data[:,:-1], y=data[:,:-1], epochs=self.epochs, 
                                     batch_size=self.batchSize, #steps_per_epoch=10,
                                     validation_data=(val, val),
                                     shuffle=True, callbacks=[self.stopping])

            self.model.save('./aeModels/autoencoder-{}'.format(round(hist.history['val_loss'][-1], 3)))
        aeOutput = self.model.get_layer(name='OutputLayer').output
        self.encoder = keras.Model(inputs=self.model.input, outputs=aeOutput)

    def predict(self, data):
        p = self.encoder.predict(x=data[:,:-1])
        return p


# Read the csv file into a pandas Dataframe as float16 to save space
def read_csv(file_path):
    with open(file_path, 'r') as f:
        column_names = next(csv.reader(f))

    dtypes = {x: np.float16 for x in column_names if x.startswith(('feature', 'target'))}
    df = pd.read_csv(file_path, dtype=dtypes, index_col=0)
    v = df.iloc[1:,2:].to_numpy()

    return df, v


def main():
    ae = AutoEncoder()
    print("Loading data...")
    # The training data is used to train your model how to predict the targets.
    training_data, train = read_csv("data/numerai_dataset_263/numerai_training_data.csv")
    # The tournament data is the data that Numerai uses to evaluate your model.
    tournament_data = read_csv("data/numerai_dataset_263/numerai_tournament_data.csv")[0]
    validation_data = tournament_data[tournament_data.data_type == "validation"]
    #printCorrelation(training_data)

    feature_names = [
        f for f in training_data.columns if f.startswith("feature")
    ]
    print("Loaded {len(feature_names)} features")

    #print('Printing features: \n', validation_data[feature_names])
    #print('Printing targets: \n', validation_data['target'])

    ae.fit(train, validation_data[feature_names].values)
    aeout = ae.predict(train)
    df = pd.DataFrame(aeout)
    df['target'] = train[:,-1:]
    corr_matrix = printCorrelation(df)
    



if __name__ == "__main__":
    main()
