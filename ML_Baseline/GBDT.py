import matplotlib.pyplot as plt
from GetData import Preprocessor_ML
from xgboost import XGBClassifier
from sklearn import metrics
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import os

if __name__ == '__main__':

    Preprocessor = Preprocessor_ML('../IEEE-CIS_Fraud_Detection')
    test_mode = ['merge_raw','raw','R-GCN','DropMajorNull']
    color_need = ['limegreen','salmon','cyan']
    plt.figure(figsize=(6, 6))
    
    for mode, color in zip(test_mode, color_need):
        X_train, X_valid, y_train, y_valid = Preprocessor.Work(mode=mode, ToObjectColumn='Encode')
        scale_pos_weight = np.sqrt((len(y_train) - sum(y_train)) / sum(y_train))

        model = GradientBoostingClassifier(random_state=10)

        model.fit(X_train,y_train)

        y_pred_prob = model.predict_proba(X_valid)[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(y_valid, y_pred_prob)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, color, label='{} Mode Val AUC = {}'.format(mode, roc_auc))
        plt.legend(loc='lower right')

    plt.title('Prediction of GBDT')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()
