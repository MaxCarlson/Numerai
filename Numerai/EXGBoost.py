import xgboost as xgb
from defines import *
from pathlib import Path

THIS_MODEL_PATH = MODEL_PATH + '/xgb/model.xgb'

class EXGBoost():
    def __init__(self):
        self.model = xgb.XGBRegressor(random_state=1, max_depth = 5, 
                                      n_estimators=200, learning_rate=0.01, 
                                      tree_method='gpu_hist', gpu_id=0)


    def fit(self, x, y):
        self.model.fit(x, y)
        self.model.save_model(THIS_MODEL_PATH)

    def predict(self, x):
        self.model = self.model.load_model(THIS_MODEL_PATH)
        p = self.model.predict(x)
        return p