import matplotlib.pyplot as plt
from sklearn import metrics
from GetData import Preprocessor_ML
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import os

if __name__ == '__main__':

    Preprocessor = Preprocessor_ML('../IEEE-CIS_Fraud_Detection')
    test_mode = ['merge_raw','raw','R-GCN','DropMajorNull']
    color_need = ['limegreen','salmon','cyan']
    plt.figure(figsize=(6, 6))
    
    for mode,color in zip(test_mode,color_need):
        X_train, X_valid, y_train, y_valid = Preprocessor.Work(mode = mode,ToObjectColumn='Encode')
        scale_pos_weight = np.sqrt((len(y_train) - sum(y_train))/sum(y_train))

        one = DecisionTreeClassifier(criterion='entropy', random_state=10, max_depth=1)

        one = one.fit(X_train, y_train)

        ada = AdaBoostClassifier(base_estimator=one,random_state=10)

        ada.fit(X_train,y_train)

        y_pred_prob = ada.predict_proba(X_valid)[:, 1]
        fpr,tpr,threshold = metrics.roc_curve(y_valid,y_pred_prob)
        roc_auc = metrics.auc(fpr,tpr)
        plt.plot(fpr, tpr, color, label='{} Mode Val AUC = {}'.format(mode,roc_auc))
        plt.legend(loc='lower right')

    plt.title('Prediction of Adaboost')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()
