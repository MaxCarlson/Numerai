# Ideas
# Maybe add difference in eras as feature?
# Add cyclical learning rates

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
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from NNetwork import NNModel
from Encoder import AutoEncoder
from Validation import validate, TARGET_NAME, PREDICTION_NAME, DIR, CURRENT_DATASET, DATASET_PATH

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

NAPI = numerapi.NumerAPI(verbosity="info")

# Download new data
NAPI.download_current_dataset(dest_path=DIR, unzip=True)

# Read the csv file into a pandas Dataframe as float16 to save space
def read_csv(file_path):
    with open(file_path, 'r') as f:
        column_names = next(csv.reader(f))

    dtypes = {x: np.float16 for x in column_names if x.startswith(('feature', TARGET_NAME))}
    df = pd.read_csv(file_path, dtype=dtypes, index_col=0)
    return df

def loadData():
    print("Loading data...")
    # The training data is used to train your model how to predict the targets.
    training_data = read_csv(DATASET_PATH + "numerai_training_data.csv")
    # The tournament data is the data that Numerai uses to evaluate your model.
    tournament_data = read_csv(DATASET_PATH  + "numerai_tournament_data.csv")
    validation_data = tournament_data[tournament_data.data_type == "validation"]
    #printCorrelation(training_data)

    feature_names = [
        f for f in training_data.columns if f.startswith("feature")
    ]
    print(f"Loaded {len(feature_names)} features")

    return training_data, tournament_data, validation_data, feature_names

def runAE(training_data, tournament_data, validation_data, feature_names):
    ae = AutoEncoder()

    #print('Printing features: \n', validation_data[feature_names])
    #print('Printing targets: \n', validation_data['target'])

    ae.fit(training_data[feature_names], validation_data[feature_names])
    aeoutTrain = ae.encode(training_data[feature_names])
    aeoutVal = ae.encode(validation_data[feature_names])

    train_corr_matrix = AutoEncoder.printCorrelation(aeoutTrain, training_data[TARGET_NAME])
    valid_corr_matrix = AutoEncoder.printCorrelation(aeoutVal, validation_data[TARGET_NAME])


def trainModel(training_data, tournament_data, validation_data, feature_names, modelName=None):
    model = NNModel()
    if not modelName:
        model.fit(training_data[feature_names], training_data[TARGET_NAME], 
                  validation_data[feature_names], validation_data[TARGET_NAME])
    return model


if __name__ == "__main__":
    training_data, tournament_data, validation_data, feature_names = loadData()

    #runAE()

    #model = trainModel(training_data, tournament_data, validation_data, feature_names, 'nn-0.693')
    model = trainModel(training_data, tournament_data, validation_data, feature_names)

    validate(training_data, tournament_data, validation_data, feature_names, model)

