import shap
import numpy as np
import pandas as pd
from defines import *
from Validation import corrAndStd, graphPerEraCorrMMC, crossValidation

# Note, we can't drop features here just based on validation data!
# Need to perform cross validation and look at common drops across all cv sets
#
# Mean Descreas Accuracy
def MDA(model, features, testSet):
    
    testSet[PREDICTION_NAME] = model.predict(testSet[features])   # predict with a pre-fitted model on an OOS validation set
    corr, std = corrAndStd(testSet)  # save base scores
    print("Base corr: ", corr)
    diff = []
    np.random.seed(42)
    for col in features:   # iterate through each features

        X = testSet.copy()
        np.random.shuffle(X[col].values)    # shuffle the a selected feature column, while maintaining the distribution of the feature
        testSet[PREDICTION_NAME] = model.predict(X[features]) # run prediction with the same pre-fitted model, with one shuffled feature
        corrX, stdX = corrAndStd(testSet)  # compare scores...
        print(col, '{:4f}'.format(corrX-corr))
        diff.append((col, corrX-corr))
    diff.sort(key=lambda x: x[1], reverse=True)
    return diff
def crossValidateMDA(model, features_names, training_data, validation_data, fraction=0.05, cv_split=4, neutral_v=0.75):
    feature_import = MDA(model, features_names, validation_data)
    # Take the top fraction features
    new_feature_names = feature_import[:-int(fraction*len(feature_import))]
    new_feature_names = [f for f, _ in new_feature_names]

    #model.fit(training_data[new_feature_names], training_data[TARGET_NAME])
    crossValidation(model, training_data, new_feature_names, split=cv_split, neuFactor=neutral_v, plot=True)



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

  