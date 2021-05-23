
import sklearn
import numpy as np
import pandas as pd
from defines import *

def neutralize(df, target=PREDICTION_NAME, by=None, proportion=1.0):
    if by is None:
        by = [x for x in df.columns if x.startswith('feature')]

    scores = df[target]
    exposures = df[by].values

    # constant column to make sure the series is completely neutral to exposures
    exposures = np.hstack((exposures, np.array([np.mean(scores)] * len(exposures)).reshape(-1, 1)))

    scores -= proportion * (exposures @ (np.linalg.pinv(exposures) @ scores.values))
    out = pd.DataFrame(scores / scores.std(), index=df.index)
    return out

def per_era_neutralization(df, feature_names, pred_name=PREDICTION_NAME):
    df[pred_name] = df.groupby("era").apply(lambda x: neutralize(x, [pred_name], feature_names))
    scaled_preds = sklearn.preprocessing.MinMaxScaler().fit_transform(df[[pred_name]])
    return scaled_preds

def norm(df):
    return (df-df.min())/(df.max()-df.min())

def addStatFeatures(data):
    fp = 'feature_'
    #fn = ['intelligence', 'charisma', 'strength', 'dexterity', 'constitution', 'wisdom']
    fn = ['constitution']#, 'constitution', 'wisdom']

    for f in fn:
        c = [col for col, _ in data.iteritems() if f in col]
        #data[fp + f + '_meanstd'] = data[c].mean(axis=1).astype(DATA_TYPE) * data[c].std(axis=1).astype(DATA_TYPE)
        #data[fp + f + '_meanvar'] = data[c].mean(axis=1).astype(DATA_TYPE) * data[c].var(axis=1).astype(DATA_TYPE)
        #data[fp + f + '_mean'] = data[c].std(axis=1).astype(DATA_TYPE)
        #data[fp + f + '_std'] = norm(data[c].std(axis=1).astype(DATA_TYPE))
        #data[fp + f + '_var'] = norm(data[c].var(axis=1).astype(DATA_TYPE))

        # TODO: Add rolling (by era) feature means/var/std etc
        
        out = data[['era'] + c].groupby("era")[c].apply(lambda x: x.mean(axis=0).astype(DATA_TYPE))
        out = data[['era'] + c].groupby("era")[c].transform(lambda x: x.std(axis=0).astype(DATA_TYPE))

        out.columns = [column + '_erastd' for column in out.columns]
        data = pd.concat([data, out], axis=1)
        #print(data)
        #cols = [[x + '_eramean'] for x in c]
        #df = pd.DataFrame(out, columns=cols)
        
    return data

def addFeatures(training_data, tournament_data):
    print('Augmenting Features...')
    
    training_data = addStatFeatures(training_data)
    tournament_data = addStatFeatures(tournament_data)

    return training_data, tournament_data


def modifyPreds(training_data, tournament_data, all_feature_names):
    print('Neutralizing Features...')
    training_data[PREDICTION_NAME] = per_era_neutralization(training_data, all_feature_names)
    tournament_data[PREDICTION_NAME] = per_era_neutralization(tournament_data, all_feature_names)