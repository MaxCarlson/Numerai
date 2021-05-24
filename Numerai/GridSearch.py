import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from defines import *
from Validation import score, load_example_data, mmc_stats

def xgbScore(_, __):
    correlations = validation_data.groupby("era").apply(score)


xgb_param_grid = {
    'n_estimators': [1500, 2500, 3500],
    'colsample_bytree': [0.1, 0.25, 0.4, 0.6],
    'max_depth': [4,5,6,7],
    'reg_alpha': [0, 0.5, 1, 5, 10],
    'reg_lambda': [1, 1.15, 1.25, 1.4],
    'subsample': [0.3, 0.5, 1]
}

def gridSearch(training_data, validation_data, feature_names, 
               param_grid=xgb_param_grid):

    # Load example preds to get MMC metrics
    validation_data = load_example_data(validation_data)

    bestParams = None
    bestScore = -np.inf

    print('Starting grid search...')
    for g in ParameterGrid(param_grid):
        model = xgb.XGBRegressor(**g, tree_method='gpu_hist', gpu_id=0)
        model.fit(training_data[feature_names], training_data[TARGET_NAME])
        validation_data[PREDICTION_NAME] = model.predict(validation_data[feature_names])

        validation_correlations = validation_data.groupby("era").apply(score)
        validation_mean = validation_correlations.mean()
        #print(f"On validation the correlation has mean {validation_mean} and "
        #    f"std {validation_correlations.std(ddof=0)}")
        #print(f"On validation the average per-era payout is {payout(validation_correlations).mean()}")
        mmc_scores, corr_scores = mmc_stats(validation_data)
        val_mmc_mean = np.mean(mmc_scores)

        newScore = validation_mean + val_mmc_mean
        if score > bestScore:
            bestParams = (*g,)
            bestScore = newScore
            print(f'Best score so far... {newScore}')
            print(f'from params {(*g,)}')

    return

