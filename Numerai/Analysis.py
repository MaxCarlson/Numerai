import shap
import pickle
import numpy as np
import pandas as pd
from defines import *
from Validation import corrAndStd, graphPerEraCorrMMC, crossValidation, crossValidation2

def interaction_filter(diffs):
    iters = [(d, c) for d, c in diffs if d.startswith('feature_interaction')]
    return iters

# Note, we can't drop features here just based on validation data!
# Need to perform cross validation and look at common drops across all cv sets
#
# Mean Descreas Accuracy
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


def crossValidateMDA(model, features_names, training_data, validation_data, 
                     mda_frac=0, cv_split=4, neutral_p=0.75, filename=None, filter_f=lambda x: x, drop_above=None):
    assert(not drop_above or not mda_frac, 'Cannot have both mda_frac and drop_above')
    feature_import = MDA(model, features_names, validation_data, filename)

    filtered_features = filter(filter_f, feature_import)

    # Take the top fraction features
    print('Dropped Features:')
    if mda_drop:
        dropPoint = -int(mda_frac * len(filtered_features))
        print(filtered_features[dropPoint:])

    # Delete features where the model improved by drop_above after removing feature in MDA
    else:
        before = set(filtered_features)
        filtered_features = [(d, c) for d, c in filtered_features if c >= drop_above]
        print(list(before - set(filtered_features)))

    new_feature_names = filtered_features[:dropPoint]
    new_feature_names = [f for f, _ in new_feature_names]

    model = crossValidation2(model, training_data, validation_data, 
                             new_feature_names, split=cv_split, neuFactor=neutral_p)
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

  