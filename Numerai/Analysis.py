import numpy as np
import pandas as pd
from Validation import score

def MDA(model, features, testSet):
    
    testSet['pred'] = model.predict(testSet[features])   # predict with a pre-fitted model on an OOS validation set
    corr, std = score(testSet)  # save base scores
    print("Base corr: ", corr)
    diff = []
    np.random.seed(42)
    for col in features:   # iterate through each features

        X = testSet.copy()
        np.random.shuffle(X[col].values)    # shuffle the a selected feature column, while maintaining the distribution of the feature
        testSet['pred'] = model.predict(X[features]) # run prediction with the same pre-fitted model, with one shuffled feature
        corrX, stdX = num.numerai_score(testSet)  # compare scores...
        print(col, corrX-corr)
        diff.append((col, corrX-corr))
        
    return diff