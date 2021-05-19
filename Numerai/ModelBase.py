import pandas as pd
import matplotlib.pyplot as plt
from defines import *

class ModelBase():
    def __init__(self):
        pass

    @staticmethod
    def plotLoss(name, history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        #plt.savefig(THIS_MODEL_PATH + 'loss.jpg')

    @staticmethod
    def printCorrelation(aeout, targets):
        aeout = pd.DataFrame(aeout)
        aeout[TARGET_NAME] = targets.values

        corr_matrix = aeout.corr()
        corr_matrix = corr_matrix[TARGET_NAME].sort_values(ascending=False)
        print(corr_matrix)
        return corr_matrix
