import xgboost as xgb
from defines import *
from pathlib import Path

THIS_MODEL_PATH = MODEL_PATH + '/xgb/'
MODEL_FILE = Path(THIS_MODEL_PATH + 'model.xgb')

class EXGBoost():
    def __init__(self, loadModel):
        self.model = xgb.XGBRegressor(random_state=1, max_depth=5, 
                                      n_estimators=3000, learning_rate=0.01,#)#, 
                                      colsample_bytree=0.25, #reg_lambda=1.5, #min_split_loss=10, 
                                      tree_method='gpu_hist', gpu_id=0)
        if loadModel:
            self.model.load_model(Path(MODEL_FILE))
    # TODO: Try early stopping?
    def fit(self, x, y):
        print('Training Model...')
        self.model.fit(x, y)
        self.model.save_model(MODEL_FILE)

    def predict(self, x):
        p = self.model.predict(x)
        return p