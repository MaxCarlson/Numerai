import xgboost as xgb
from defines import *
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from Validation import valid_metrics

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

    def eraFit(self, training_data, validation_data, feature_names,
               proportion=0.5, trees_per_step=10, num_iters=200): #default: 0.5, 10, 200
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

        # Era split determines how often we train on all the data
        # vs just the worst era data
        eraSplit = 5
        epochPrint = 1
        sampling = 0

        # Best so far, proportion = 0.5, eraSplit=5, epochs=24, lr=0.01, depth=5, colsample_bytree=0.1, reg_lambda=1.5, alpha=1, 
        # erasplit=5 was similar, epochs=33
        # W/ augment vcorr=0.28 w/0 0.275


        self.model = xgb.XGBRegressor(random_state=self.rs, max_depth=5, learning_rate=0.01, 
                                      n_estimators=trees_per_step, colsample_bytree=0.1, 
                                      reg_lambda=1.5, alpha=1)#,
                                      #tree_method='gpu_hist', gpu_id=0)

        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(10, 5.5)
        l1, = ax1.plot([],[])
        l2, = ax2.plot([],[])
        l3, = ax3.plot([],[])

        self.model.fit(training_data[feature_names], training_data[TARGET_NAME])
        new_df = training_data.copy()
        new_vf = validation_data.copy()
        for i in range(num_iters):
            print(f"iteration {i}")
            # score each era
            #print("predicting on train")
            new_df[PREDICTION_NAME] = self.model.predict(training_data[feature_names])
            era_scores = pd.Series(index=new_df["era"].unique())
            if i % eraSplit == 0:
                print("getting per era scores")
                for era in new_df["era"].unique():
                    era_df = new_df[new_df["era"] == era]
                    era_scores[era] = spearmanr(era_df[PREDICTION_NAME], era_df[TARGET_NAME])
                era_scores.sort_values(inplace=True)
                worst_eras = era_scores[era_scores <= era_scores.quantile(proportion)].index
                worst_df = new_df[new_df["era"].isin(worst_eras)]
                if sampling:
                    worst_df = worst_df.append(new_df[new_df['era'].isin(
                        era_scores[era_scores > era_scores.quantile(proportion)])].sample(frac=sampling))
            else:
                worst_df = new_df
            #print(list(worst_eras))
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
            self.model.n_estimators += trees_per_step #* (eraSplit if i % eraSplit == 0 else 1)
            booster = self.model.get_booster()
            #print("fitting on worst eras")
            self.model.fit(worst_df[feature_names], worst_df[TARGET_NAME], xgb_model=booster)

            if i % epochPrint == 0:
                new_vf[PREDICTION_NAME] = self.model.predict(new_vf[feature_names])
                vcorr, vsharpe, vdown = valid_metrics(new_vf)
                
                l1.set_xdata(np.append(l1.get_xdata(), i))
                l2.set_xdata(np.append(l2.get_xdata(), i))
                l3.set_xdata(np.append(l3.get_xdata(), i))
                l1.set_ydata(np.append(l1.get_ydata(), vcorr.mean()))
                l2.set_ydata(np.append(l2.get_ydata(), vsharpe))
                l3.set_ydata(np.append(l3.get_ydata(), vdown))
                ax1.relim()
                ax2.relim()
                ax3.relim()
                ax1.autoscale_view()
                ax2.autoscale_view()
                ax3.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
        return self.model

    def predict(self, x):
        p = self.model.predict(x)
        return p