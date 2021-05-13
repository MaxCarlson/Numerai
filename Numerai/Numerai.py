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


# Download new data
DIR = "./data/"
NAPI.download_current_dataset(dest_path=DIR, unzip=True)

def printCorrelation(df):
    corr_matrix = df.corr()
    corr_matrix = corr_matrix['target'].sort_values(ascending=False, method='spearman')
    print(corr_matrix)
    return corr_matrix

class AutoEncoder():
    
    lr = 0.009
    epochs = 3
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
        # Note: Current best:
        # 310, 256, 196, 128: lr=0.009, decay_steps=6800, decay_rate=0.95
        # act=tanh, out=sigmoid, loss=binary_entropy, optimizer=adam
        # Epochs=3, val_loss=0.606
        # Features > 0.01: 0.0148, 0.0131

        self.model = models.Sequential()
        self.model.add(layers.Dense(310, activation='tanh'))
        self.model.add(layers.Dense(256, activation='tanh'))
        self.model.add(layers.Dense(196, activation='tanh'))
        #self.model.add(layers.Dense(128, activation='tanh'))
        #self.model.add(layers.Dense(64, activation='tanh'))
        self.model.add(layers.Dense(128, activation='tanh', name='OutputLayer'))
        #self.model.add(layers.Dense(64, activation='tanh'))
        #self.model.add(layers.Dense(128, activation='tanh'))
        self.model.add(layers.Dense(196, activation='tanh'))
        self.model.add(layers.Dense(256, activation='tanh'))
        self.model.add(layers.Dense(310, activation='sigmoid'))
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.decay), 
              #loss=keras.losses.mean_squared_error,
              loss=keras.losses.binary_crossentropy,
              metrics=[])

    def fit(self, data):
        history = self.model.fit(x=data[:,:-1], y=data[:,:-1], epochs=self.epochs, 
                                 batch_size=self.batchSize, validation_split=0.1, shuffle=True, 
                                 callbacks=[self.stopping])
        
        # Testing 
        #history = self.model.fit(x=data[:,:-1], y=data[:,:-1], epochs=1, 
        #                         batch_size=self.batchSize, shuffle=True, 
        #                         steps_per_epoch=len(data)//32//self.batchSize)

        # Build a model to produce the compressed output from the autoencoder
        aeOutput = self.model.get_layer(name='OutputLayer').output
        aeModel  = keras.Model(inputs=self.model.input, outputs=aeOutput)
        p = aeModel.predict(x=data[:,:-1])

        return p

    def findTopOutputs(self, corr_matrix):
        idxs = []
        for i in range(1, len(corr_matrix)):
            if corr_matrix.iloc[i, 1] < self.featureThreshold:
                break
            idxs.append(corr_matrix.iloc[i, 0])
            
        return i, idxs

    def save(self, corr_matrix):
        greaterThanBase, idxs = self.findTopOutputs(corr_matrix)
        self.model.save(filepath=('./aeModels/autoencoder-{}-%s.h5'.format(greaterThanBase, ) % ','.join(idxs))
                        , overwrite=False, include_optimizer=False)



# Read the csv file into a pandas Dataframe as float16 to save space
def read_csv(file_path):
    with open(file_path, 'r') as f:
        column_names = next(csv.reader(f))

    dtypes = {x: np.float16 for x in column_names if x.startswith(('feature', 'target'))}
    df = pd.read_csv(file_path, dtype=dtypes, index_col=0)
    v = df.iloc[1:,2:].to_numpy()

    # Memory constrained? Try this instead (slower, but more memory efficient)
    # see https://forum.numer.ai/t/saving-memory-with-uint8-features/254
    # dtypes = {f"target": np.float16}
    # to_uint8 = lambda x: np.uint8(float(x) * 4)
    # converters = {x: to_uint8 for x in column_names if x.startswith('feature')}
    # df = pd.read_csv(file_path, dtype=dtypes, converters=converters)

    return df, v


def main():
    ae = AutoEncoder()
    print("Loading data...")
    # The training data is used to train your model how to predict the targets.
    training_data, train = read_csv("data/numerai_dataset_263/numerai_training_data.csv")
    # The tournament data is the data that Numerai uses to evaluate your model.
    #tournament_data = read_csv("data/numerai_dataset_263/numerai_tournament_data.csv")
    #printCorrelation(training_data)

    feature_names = [
        f for f in training_data.columns if f.startswith("feature")
    ]
    print(f"Loaded {len(feature_names)} features")

    aeout = ae.fit(train)
    df = pd.DataFrame(aeout)
    df['target'] = train[:,-1:]
    corr_matrix = printCorrelation(df)
    ae.save(corr_matrix)
    



if __name__ == "__main__":
    main()
