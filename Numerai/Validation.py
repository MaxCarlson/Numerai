import sklearn
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from defines import *

# Submissions are scored by spearman correlation
def correlation(predictions, targets):
    ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]

# convenience method for scoring
def score(df):
    return correlation(df[PREDICTION_NAME], df[TARGET_NAME])

def corrAndStd(df):
    corrs = df.groupby("era").apply(score)
    return corrs.mean(), corrs.std(ddof=0)

# Payout is just the score cliped at +/-25%
def payout(scores):
    return scores.clip(lower=-0.25, upper=0.25)

def validate(training_data, tournament_data, validation_data, 
             feature_names, model, savePreds=False):


    # Check the per-era correlations on the training set (in sample)
    if 'era' in training_data.columns:
        train_correlations = training_data.groupby("era").apply(score)
        print(f"On training the average per-era payout is {payout(train_correlations).mean()}")
    else:
        train_correlations = score(training_data)
    print(f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std(ddof=0)}")

    """Validation Metrics"""
    # Check the per-era correlations on the validation set (out of sample)
    validation_correlations = validation_data.groupby("era").apply(score)
    print(f"On validation the correlation has mean {validation_correlations.mean()} and "
          f"std {validation_correlations.std(ddof=0)}")
    print(f"On validation the average per-era payout is {payout(validation_correlations).mean()}")
    
    # Check the "sharpe" ratio on the validation set
    validation_sharpe = validation_correlations.mean() / validation_correlations.std(ddof=0)
    print(f"Validation Sharpe: {validation_sharpe}")

    print("checking max drawdown...")
    rolling_max = (validation_correlations + 1).cumprod().rolling(window=100,
                                                                  min_periods=1).max()
    daily_value = (validation_correlations + 1).cumprod()
    max_drawdown = -((rolling_max - daily_value) / rolling_max).max()
    print(f"max drawdown: {max_drawdown}")

    # Check the feature exposure of your validation predictions
    feature_exposures = validation_data[feature_names].apply(lambda d: correlation(validation_data[PREDICTION_NAME], d),
                                                             axis=0)
    max_per_era = validation_data.groupby("era").apply(
        lambda d: d[feature_names].corrwith(d[PREDICTION_NAME]).abs().max())
    max_feature_exposure = max_per_era.mean()
    print(f"Max Feature Exposure: {max_feature_exposure}")

    # Check feature neutral mean
    print("Calculating feature neutral mean...")
    feature_neutral_mean = get_feature_neutral_mean(validation_data)
    print(f"Feature Neutral Mean is {feature_neutral_mean}")

    # Load example preds to get MMC metrics
    validation_data, validation_example_preds = load_example_data(validation_data)

    print("calculating MMC stats...")
    # MMC over validation
    mmc_scores, corr_scores = mmc_stats(validation_data)

    val_mmc_mean = np.mean(mmc_scores)
    val_mmc_std = np.std(mmc_scores)
    val_mmc_sharpe = val_mmc_mean / val_mmc_std
    corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
    corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)
    corr_plus_mmc_mean = np.mean(corr_plus_mmcs)
    corr_plus_mmc_sharpe_diff = corr_plus_mmc_sharpe - validation_sharpe

    print(
        f"MMC Mean: {val_mmc_mean}\n"
        f"Corr Plus MMC Sharpe:{corr_plus_mmc_sharpe}\n"
        f"Corr Plus MMC Diff:{corr_plus_mmc_sharpe_diff}"
    )

    # Check correlation with example predictions
    full_df = pd.concat([validation_example_preds, validation_data[PREDICTION_NAME], validation_data["era"]], axis=1)
    full_df.columns = ["example_preds", "prediction", "era"]
    per_era_corrs = full_df.groupby('era').apply(lambda d: correlation(unif(d["prediction"]), unif(d["example_preds"])))
    corr_with_example_preds = per_era_corrs.mean()
    print(f"Corr with example preds: {corr_with_example_preds}")

    if savePreds:
        print('Saving Submissions...')
        tournament_data[PREDICTION_NAME].to_csv("submission.csv", header=True)

def load_example_data(validation_data):
    example_preds = pd.read_csv(DATASET_PATH + "example_predictions.csv").set_index("id")["prediction"]
    validation_example_preds = example_preds.loc[validation_data.index]
    validation_data["ExamplePreds"] = validation_example_preds
    return validation_data, validation_example_preds

def mmc_stats(validation_data):
    # MMC over validation
    mmc_scores = []
    corr_scores = []
    for _, x in validation_data.groupby("era"):
        series = neutralize_series(pd.Series(unif(x[PREDICTION_NAME])),
                                   pd.Series(unif(x["ExamplePreds"])))
        mmc_scores.append(np.cov(series, x[TARGET_NAME])[0, 1] / (0.29 ** 2))
        corr_scores.append(correlation(unif(x[PREDICTION_NAME]), x[TARGET_NAME]))
    return mmc_scores, corr_scores


# to neutralize a column in a df by many other columns on a per-era basis
def neutralize(df,
               columns,
               extra_neutralizers=None,
               proportion=1.0,
               normalize=True,
               era_col="era"):
    # need to do this for lint to be happy bc [] is a "dangerous argument"
    if extra_neutralizers is None:
        extra_neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        print(u, end="\r")
        df_era = df[df[era_col] == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (pd.Series(x).rank(method="first").values - .5) / len(x)
                scores2.append(x)
            scores = np.array(scores2).T
            extra = df_era[extra_neutralizers].values
            exposures = np.concatenate([extra], axis=1)
        else:
            exposures = df_era[extra_neutralizers].values

        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))

        scores /= scores.std(ddof=0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed),
                        columns=columns,
                        index=df.index)


# to neutralize any series by any other series
def neutralize_series(series, by, proportion=1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures,
         np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

    correction = proportion * (exposures.dot(
        np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized


def unif(df):
    x = (df.rank(method="first") - 0.5) / len(df)
    return pd.Series(x, index=df.index)


def get_feature_neutral_mean(df):
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df.loc[:, "neutral_sub"] = neutralize(df, [PREDICTION_NAME],
                                          feature_cols)[PREDICTION_NAME]
    scores = df.groupby("era").apply(
        lambda x: correlation(x["neutral_sub"], x[TARGET_NAME])).mean()
    return np.mean(scores)

#################################################################################
## User Stuff                                                                  ##
#################################################################################

def graphPerEraCorrSharpe(data, multi=1, mmc_mult=0.5):
    corrs = data.groupby("era").apply(score)
    #sharp = corrs.mean() / corrs.std(ddof=0)
    load_example_data(data)
    mmcs, corr_scores = mmc_stats(data)
    X = [x for x in range(len(corrs))]

    def calcPayouts(corr_mult, mmc_mult):
        payout = []
        base = 1.0
        for corr, mmc in zip(corrs, mmcs):
            base += base * multi * (corr + mmc * mmc_mult)
            payout.append(base)
        return payout

    fig, (ax1, ax3, ax5) = plt.subplots(1, 3)
    def graph(ax, data, color, xlabel, ylabel):
        ax.set_xlabel(xlabel)
        ax.plot(X, data, color=color)
        ax.set_ylabel(ylabel, color=color)
        ax.tick_params(axis='y', labelcolor=color)
    
    # Plot correlation and payout by era
    graph(ax1, corrs, 'tab:blue', 'Era', 'corr')
    ax2 = ax1.twinx()
    graph(ax2, calcPayouts(1, 0), 'tab:red', 'Era', 'Payout over time')

    # Plot mmc and mmc payout by era
    graph(ax3, mmcs, 'tab:blue', 'Era', 'mmc')
    ax4 = ax3.twinx()
    graph(ax4, calcPayouts(0, 1), 'tab:red', 'Era', 'Payout over time')

    # Plot corr+mmc payout
    colormap = np.array(['b', 'g', 'r', 'm'])
    patches = [mpatches.Patch(color=c, label=l) for c,l in zip(
        colormap, ['mmc=0', 'mmc=0.5', 'mmc=1', 'mmc=2'])]
    ax5.legend(handles=patches)
    graph(ax5, calcPayouts(1, 0), colormap[0],'Era', 'Payout')
    graph(ax5, calcPayouts(1, 0.5), colormap[1],'Era', 'Payout')
    graph(ax5, calcPayouts(1, 1), colormap[2], 'Era', 'Payout')
    graph(ax5, calcPayouts(1, 2), colormap[3], 'Era', 'Payout')

    #fig.tight_layout()
    ax1.set_title(f'corr/payout')
    ax3.set_title(f'mmc/mmc payout mmc={1}')
    ax5.set_title(f'mmc+corr payout mmc={mmc_mult}')
    plt.show()