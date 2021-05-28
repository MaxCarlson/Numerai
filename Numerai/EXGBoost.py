import xgboost as xgb
from defines import *
from pathlib import Path

THIS_MODEL_PATH = MODEL_PATH + '/xgb/'
MODEL_FILE = Path(THIS_MODEL_PATH + 'model.xgb')

class EXGBoost():
    def __init__(self, loadModel):
        self.model = xgb.XGBRegressor(random_state=1, max_depth=5, objective='reg:squarederror',
                        #sampling_method='uniform', subsample=0.75,
                        n_estimators=3000, learning_rate=0.01,
                        colsample_bytree=0.25, reg_lambda=1.2, alpha=0, #gamma=0.01, 
                        tree_method='gpu_hist', gpu_id=0)
        if loadModel:
            self.model.load_model(Path(MODEL_FILE))
    # TODO: Try early stopping?
    def fit(self, x, y, xv, yv, saveModel=False):
        self.model.fit(x, y)#, eval_set=[(x, y), (xv, yv)], early_stopping_rounds=1000)
        if saveModel:
            self.model.save_model(MODEL_FILE)

    def predict(self, x):
        p = self.model.predict(x)
        return p