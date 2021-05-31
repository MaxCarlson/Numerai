import xgboost as xgb
from defines import *
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

THIS_MODEL_PATH = MODEL_PATH + '/xgb/'
MODEL_FILE = Path(THIS_MODEL_PATH + 'model.xgb')

class EXGBoost():
    rs = 1
    def __init__(self, loadModel):
        self.model = xgb.XGBRegressor(random_state=self.rs, max_depth=5, objective='reg:squarederror',
                        #sampling_method='uniform', subsample=0.75,
                        n_estimators=3000, learning_rate=0.01,
                        colsample_bytree=0.25, reg_lambda=1.2, alpha=0, #gamma=0.01, 
                        tree_method='gpu_hist', gpu_id=0)
        if loadModel:
            self.model.load_model(Path(MODEL_FILE))
    # TODO: Try early stopping?
    def fit(self, x, y, xv=None, yv=None, saveModel=False):
        self.model.fit(x, y)#, eval_set=[(x, y), (xv, yv)], early_stopping_rounds=1000)
        if saveModel:
            self.model.save_model(MODEL_FILE)

    def eraFit(self, x, y, era_col, proportion=0.5, trees_per_step=10, num_iters=100): #default: 0.5, 10, 200
        def spearmanr(target, pred):
            return np.corrcoef(target, pred.rank(pct=True, method="first"))[0, 1]
        def ar1(x):
            return np.corrcoef(x[:-1], x[1:])[0,1]

        def autocorr_penalty(x):
            n = len(x)
            p = ar1(x)
            return np.sqrt(1 + 2*np.sum([((n - i)/n)*p**i for i in range(1,n)]))

        def smart_sharpe(x):
            return np.mean(x)/(np.std(x, ddof=1)*autocorr_penalty(x))

        features = x.columns
        self.model = xgb.XGBRegressor(random_state=self.rs, max_depth=5, learning_rate=0.01, 
                                      n_estimators=trees_per_step, colsample_bytree=0.07, 
                                      reg_lambda=1.5, alpha=1,
                                      tree_method='gpu_hist', gpu_id=0)
        self.model.fit(x, y)
        new_df = x.copy()
        new_df[TARGET_NAME] = y
        new_df["era"] = era_col
        for i in range(num_iters):
            print(f"iteration {i}")
            # score each era
            print("predicting on train")
            preds = self.model.predict(x)
            new_df["pred"] = preds
            era_scores = pd.Series(index=new_df["era"].unique())
            print("getting per era scores")
            for era in new_df["era"].unique():
                era_df = new_df[new_df["era"] == era]
                era_scores[era] = spearmanr(era_df["pred"], era_df["target"])
            era_scores.sort_values(inplace=True)
            worst_eras = era_scores[era_scores <= era_scores.quantile(proportion)].index
            #print(list(worst_eras))
            worst_df = new_df[new_df["era"].isin(worst_eras)]
            #era_scores.sort_index(inplace=True)
            #era_scores.plot(kind="bar")
            #print("performance over time")
            ##plt.show()
            #print("autocorrelation")
            #print(ar1(era_scores))
            #print("mean correlation")
            #print(np.mean(era_scores))
            #print("sharpe")
            #print(np.mean(era_scores)/np.std(era_scores))
            #print("smart sharpe")
            #print(smart_sharpe(era_scores))
            self.model.n_estimators += trees_per_step
            booster = self.model.get_booster()
            #print("fitting on worst eras")
            self.model.fit(worst_df[features], worst_df["target"], xgb_model=booster)
        return self.model

    def predict(self, x):
        p = self.model.predict(x)
        return p