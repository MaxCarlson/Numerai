import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from defines import *
from Validation import score, load_example_data, mmc_stats

xgb_param_grid = {
    #'n_estimators': [2500, 3000, 3500],
    #'learning_rate' : [0.001, 0.01],
    #'max_depth': [4,5,6],
    #'objective': ['reg:squarederror', 'reg:squaredlogerror', 'reg:logistic', 'reg:pseudohubererror']
    'colsample_bytree': [0.15, 0.25],
    'reg_alpha': [0, 7],
    'reg_lambda': [1, 1.25, 1.5],
    'subsample': [0.5, 0.85],
    'max_depth': [5], 
    'n_estimators': [3000], 
    'learning_rate': [0.01] 
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
        print(f'Score: {newScore} for {g}')
        if newScore > bestScore:
            bestParams = (*g,)
            bestScore = newScore
            print(f'\nBest score so far... {newScore}')
            print(f'from params {g}\n')

    return

