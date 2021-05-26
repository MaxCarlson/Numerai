
import sklearn
import numpy as np
import pandas as pd
from defines import *
import smote_variants as smote
from sklearn import preprocessing

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

def per_era_neutralization(df, feature_names, prop=0.5, pred_name=PREDICTION_NAME):
    df[pred_name] = df.groupby("era").apply(lambda x: neutralize(x, [pred_name], feature_names, prop))
    scaled_preds = sklearn.preprocessing.MinMaxScaler().fit_transform(df[[pred_name]])
    return scaled_preds

def norm(df):
    return (df-df.min())/(df.max()-df.min())


def interactions(training_data, tournament_data, feature_names):
    p_features = [['dexterity6', 'charisma63', 'dexterity7', 'wisdom35']]
    p_features = [['feature_' + x for x in l] for l in p_features]

    for features in p_features:
        interactions = preprocessing.PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        interactions.fit(training_data[features], training_data[TARGET_NAME])
        train_interactions      = pd.DataFrame(interactions.transform(training_data[features]), index=training_data.index)
        tournament_interactions = pd.DataFrame(interactions.transform(tournament_data[features]), index=tournament_data.index)

        training_data   = pd.concat([training_data, train_interactions],axis=1)
        tournament_data = pd.concat([tournament_data, tournament_interactions],axis=1)
    
    return training_data, tournament_data

def addStatFeatures(data):
    f = 'feature_'
    groups = ['intelligence', 'charisma', 'strength', 'dexterity', 'constitution', 'wisdom']
    #groups = ['wisdom']#, 'constitution', 'wisdom']

    for group in groups:
        c = [col for col, _ in data.iteritems() if f in col]
        #data[f + group + '_meanstd'] = data[c].mean(axis=1).astype(DATA_TYPE) * data[c].std(axis=1).astype(DATA_TYPE)
        #data[f + group + '_meanvar'] = data[c].mean(axis=1).astype(DATA_TYPE) * data[c].var(axis=1).astype(DATA_TYPE)
        #data[f + group + '_mean'] = data[c].std(axis=1).astype(DATA_TYPE)
        #data[f + group + '_std'] = norm(data[c].std(axis=1).astype(DATA_TYPE))
        #data[f + group + '_var'] = norm(data[c].var(axis=1).astype(DATA_TYPE))

        #data[f + group + '_skew'] = data[c].skew(axis=1).astype(DATA_TYPE)

        # TODO: Find features that have high correlation between stocks 
        # in a specific era and create new stat features from them!
        
        #out = data[['era'] + c].groupby("era")[c].apply(lambda x: x.mean(axis=0).astype(DATA_TYPE))

        # Create per-era statistical features
        #out = data[['era'] + c].groupby("era")[c].transform(lambda x: x.std(axis=0).astype(DATA_TYPE))
        ##out = norm(out)
        #out.columns = [column + '_erastd' for column in out.columns]
        #data = pd.concat([data, out], axis=1)

        
    return data

def applySmote(data, feature_names):
    oversampler = smote.MulticlassOversampling(smote.polynom_fit_SMOTE(random_state=1))
    X, Y = oversampler.sample(data[feature_names], data[TARGET_NAME])
    print(f'New data after smote size {X.shape}')
    data = pd.DataFrame(X, columns=feature_names, dtype=DATA_TYPE)
    data[TARGET_NAME] = Y
    return data



def addFeatures(training_data, tournament_data, feature_names):
    print('Augmenting Features...')
    
    training_data = addStatFeatures(training_data)
    tournament_data = addStatFeatures(tournament_data)
    #training_data = applySmote(training_data, feature_names)

    training_data, tournament_data = interactions(training_data, tournament_data, feature_names)

    return training_data, tournament_data


def modifyPreds(training_data, tournament_data, all_feature_names, f_prop=0.5):
    print('Neutralizing Features...')
    training_data[PREDICTION_NAME] = per_era_neutralization(training_data, all_feature_names, f_prop)
    tournament_data[PREDICTION_NAME] = per_era_neutralization(tournament_data, all_feature_names, f_prop)