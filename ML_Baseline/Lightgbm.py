import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn import metrics
from GetData import Preprocessor_ML
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import os

def LGB_test(train_x,train_y,test_x,test_y):
    from multiprocessing import cpu_count
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=2, n_estimators=800,max_features = 140, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50,random_state=None,n_jobs=cpu_count()-1,
        num_iterations = 800 #迭代次数
    )
    clf.fit(train_x, train_y,eval_set=[(train_x, train_y),(test_x,test_y)],eval_metric='auc',early_stopping_rounds=100)

    return clf,clf.best_score_[ 'valid_1']['auc']
  
  if __name__ == '__main__':

    Preprocessor = Preprocessor_ML('../IEEE-CIS_Fraud_Detection')
    test_mode = ['merge_raw','raw','R-GCN','DropMajorNull']
    color_need = ['limegreen','salmon','cyan']
    plt.figure(figsize=(6, 6))

    for mode,color in zip(test_mode,color_need):
        X_train, X_valid, y_train, y_valid = Preprocessor.Work(mode = mode,ToObjectColumn='Encode')
        scale_pos_weight = np.sqrt((len(y_train) - sum(y_train))/sum(y_train))

        print('Starting training...')
        # 模型训练
        model, auc = LGB_test(X_train,y_train,X_valid,y_valid)

        y_pred_prob = model.predict_proba(X_valid)[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(y_valid, y_pred_prob)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, color, label='{} Mode Val AUC = {}'.format(mode, roc_auc))
        plt.legend(loc='lower right')

    plt.title('Prediction of lightgbm')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()
