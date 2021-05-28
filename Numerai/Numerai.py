# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Feature Engineering                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Use autoencoder for feature engineering!
# Try count-encoding, switch feature value to it's respective frequency of occurance in that feature
# Some sort of feature selection search?
# Take only the most highly correlated features? Univariate Feature Selection?
# Mean/Median/etc of feature catagories

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Best ideas                                                  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Group datasets by eras. Train on all eras, then perform boosting on more difficult eras
# (eras with low target correlation)!!!
# Feature neutralization (try neutralizing on per era basis)
# Train encoder with custom loss function set to maximize output correlation to target?
# Boost NN model with scikit-learn AdaBoostRegressor
# Add NN outputs into input data for XGBoost
# Train ensamble using feature groups?

# TODO
# Switch to pd.HDFStore('numera_training_data.h5')

import os
import csv
import warnings
import numerapi
import numpy as np
import pandas as pd
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from defines import *
from DataAugment import addFeatures, modifyPreds
from NNetwork import NNModel
from Encoder import AutoEncoder
from Validation import validate, neutralize_series, crossValidation
from EXGBoost import EXGBoost
from GridSearch import gridSearch
from Analysis import applyAnalysis

warnings.filterwarnings('ignore')


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

    dtypes = {x: DATA_TYPE for x in column_names if x.startswith(('feature', TARGET_NAME))}
    
    df = pd.read_csv(file_path, dtype=dtypes, index_col=0)
    return df

def loadData(path=DATASET_PATH):

    print(f"Loading dataset {DATASET_PATH}...")
    if not os.path.isfile(path + "data.h5"):
        print('Saving new dataset as hdf5...')
        training_data = read_csv(path + "numerai_training_data.csv")
        tournament_data = read_csv(path  + "numerai_tournament_data.csv")

        training_data.to_hdf(path + "data.h5", key='training')
        tournament_data.to_hdf(path + "data.h5", key='tournament')
    else:
        training_data = pd.read_hdf(path + 'data.h5', key='training')
        tournament_data = pd.read_hdf(path + "data.h5", key='tournament')


    #printCorrelation(training_data)
    o_feature_names = [ #['era']+
        f for f in training_data.columns if f.startswith("feature")]
    print(f"Loaded {len(o_feature_names)} features")

    training_data, tournament_data = addFeatures(training_data, tournament_data, o_feature_names)

    feature_names = [ #['era']+
        f for f in training_data.columns if f.startswith("feature")]
    print('Added {} features'.format(len(feature_names) - len(o_feature_names)))

    validation_data = tournament_data[tournament_data.data_type == "validation"]
    return training_data, tournament_data, validation_data, feature_names, o_feature_names

def runAE(training_data, tournament_data, validation_data, feature_names, 
          saveData=False, modelName=None, printCorr=False):
    ae = AutoEncoder()
    if modelName:
        ae.load(modelName)
    else:
        ae.fit(training_data[feature_names], validation_data[feature_names])

    #print('Printing features: \n', validation_data[feature_names])
    #print('Printing targets: \n', validation_data['target'])
    if printCorr:
        aeoutTrain = ae.encode(training_data[feature_names])
        aeoutVal = ae.encode(validation_data[feature_names])
        train_corr_matrix = AutoEncoder.printCorrelation(aeoutTrain, training_data[TARGET_NAME])
        valid_corr_matrix = AutoEncoder.printCorrelation(aeoutVal, validation_data[TARGET_NAME])

    if saveData: 
        ae.saveData(training_data, tournament_data, feature_names, '0.423')
    stop=True



def trainModel(training_data, tournament_data, validation_data, feature_names, modelName=None):
    model = NNModel()
    if not modelName:
        model.fit(training_data[feature_names], training_data[TARGET_NAME], 
                  validation_data[feature_names], validation_data[TARGET_NAME])
    return model


if __name__ == "__main__":
    alteredData=False

    if alteredData:
        training_data, tournament_data, validation_data, feature_names, o_features_names = loadData('models/aeModels/autoencoder-0.423/')
    else:
        training_data, tournament_data, validation_data, feature_names, o_features_names = loadData()


    #gridSearch(training_data, validation_data, feature_names)
    #model = NNModel('-0.051')
    model = EXGBoost(loadModel=False)

    crossValidation(model, training_data, feature_names, split=4, neuFactor=0, plot=True)
    
    print('Training Model...')
    model.fit(training_data[feature_names], training_data[TARGET_NAME], 
            validation_data[feature_names], validation_data[TARGET_NAME], saveModel=True)
    

    #runAE(training_data, tournament_data, validation_data, feature_names)
    #runAE(training_data, tournament_data, validation_data, feature_names, True, '0.423')

    #tp = model.predict(training_data[feature_names])#, DATASET_PATH)
    #training_data['nnpred'] = tp
    #tp = model.predict(tournament_data[feature_names])#, DATASET_PATH)
    #tournament_data['nnpred'] = tp
    #feature_names += ['nnpred']
    #validation_data = tournament_data[tournament_data.data_type == "validation"]

    #model = trainXGBoost(training_data, tournament_data, validation_data, feature_names, loadModel=False)
    #model = trainXGBoost(training_data, tournament_data, validation_data, feature_names, loadModel=True)

    print('Starting Predictions...')
    training_data[PREDICTION_NAME] = model.predict(training_data[feature_names])
    tournament_data[PREDICTION_NAME] = model.predict(tournament_data[feature_names])
    print('Predictions done...')

    modifyPreds(training_data, tournament_data, feature_names, f_prop=0.75)
    validation_data[PREDICTION_NAME] = tournament_data[PREDICTION_NAME]



    # Load non manipulated data for validation purposes
    if alteredData:
        pass
        #atraining_data, atournament_data, validation_data, afeature_names = loadData()
        #validate(atraining_data, atournament_data, validation_data, 
        #         afeature_names, model, training_data, tournament_data, 
        #         feature_names, savePreds=False)
    else:

        # Note: we're not looking at feature exposure for new features here?
        validate(training_data, tournament_data, validation_data, 
                 o_features_names, model, savePreds=True)
    
    applyAnalysis(model, feature_names, validation_data)

