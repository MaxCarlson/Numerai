import shap
import pickle
import numpy as np
import pandas as pd
from defines import *
from Validation import *
#from Validation import corrAndStd, graphPerEraCorrMMC, crossValidation, crossValidation2

def interaction_filter(diffs):
    iters = [(d, c) for d, c in diffs if d.startswith('feature_interaction')]
    return iters

# Note, we can't drop features here just based on validation data!
# Need to perform cross validation and look at common drops across all cv sets
#
# Mean Decrease Accuracy
#
# TODO: Apply MDA but using sharpe ratio & vcor together so we don't remove feature that prevent losses!
# TODO: Apply MDA but using sharpe ratio & vcor together so we don't remove feature that prevent losses!
#
def MDA(model, features, testSet, filename=None):

    if filename:
        try:
            with open(filename, 'rb') as fp:
                diff = pickle.load(fp)
                print(f'Loaded file {filename} for MDA')
                return diff
        except:
            NameError

    testSet[PREDICTION_NAME] = model.predict(testSet[features])   # predict with a pre-fitted model on an OOS validation set
    corr, std = corrAndStd(testSet)  # save base scores
    print("Base corr: ", corr)
    diff = []
    for col in features:   # iterate through each features

        X = testSet.copy()
        np.random.shuffle(X[col].values)    # shuffle the a selected feature column, while maintaining the distribution of the feature
        testSet[PREDICTION_NAME] = model.predict(X[features]) # run prediction with the same pre-fitted model, with one shuffled feature
        corrX, stdX = corrAndStd(testSet)  # compare scores...
        print(col, '{:4f}'.format(corrX-corr))
        diff.append((col, corrX-corr))
    diff.sort(key=lambda x: x[1])

    if filename:
        with open(filename, 'wb') as fp:
            pickle.dump(diff, fp)

    return diff

def applyMDA(model, features_names, training_data, validation_data, 
                     cv_split=4, neutral_p=0.75, filename=None, filter_f=lambda x: x, mda_frac=None, drop_above=None):
    assert((drop_above!=None) ^ (mda_frac!=None), 'Cannot have both mda_frac and drop_above or neither')
    feature_import = MDA(model, features_names, validation_data, filename)

    print('Starting Cross Validation MDA...')
    filtered_features = filter_f(feature_import)

    # Take the top fraction features
    print('Dropped Features:')
    if mda_frac:
        dropPoint = -int(mda_frac * len(filtered_features))
        print(filtered_features[dropPoint:])
        new_feature_names = filtered_features[:dropPoint]

    # Delete features where the model improved by drop_above after removing feature in MDA
    else:
        filtered_features = [(d, c) for d, c in filtered_features if c >= drop_above]
        print(filtered_features)
        new_feature_names = list(set(feature_import) - set(filtered_features))

    new_feature_names = [f for f, _ in new_feature_names]

    return new_feature_names


def crossValidateMDA(model, features_names, training_data, validation_data, validation_type,
                     cv_split=4, neutral_p=0.75, filename=None, filter_f=lambda x: x, mda_frac=None, 
                     drop_above=None):

    new_feature_names = applyMDA(model, features_names, training_data, validation_data, cv_split, 
             neutral_p, filename, filter_f, mda_frac, drop_above)
    model = crossValidation2(model, training_data, validation_data, 
                             new_feature_names, split=cv_split, neuFactor=neutral_p, valid_type=validation_type)
    return model, new_feature_names



def applyShap(model, feature_names, dataset):
    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    explainer = shap.Explainer(model.model)
    shap_values = explainer(dataset[feature_names])

    # visualize the first prediction's explanation

    print(shap_values)

    #shap.plots.waterfall(shap_values[0])
    shap.plots.beeswarm(shap_values, max_display=30)


def applyAnalysis(model, feature_names, dataset):
    #MDA(model, feature_names, dataset)
    #applyShap(model, feature_names, dataset)

    graphPerEraCorrMMC(dataset)

  