# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 17:03:42 2020

@author: gilta
"""
import os.path as pth
import yaml
with open('../config.yaml', 'r') as fp:
    config = yaml.load(fp, yaml.FullLoader)
path = pth.dirname(pth.abspath(__file__))[:-3] + '/'
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score as AUC_score

def main():
    df = pd.read_csv(path + config['training_data'])

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


    ## PARAMS for XGBoost :
    num_boost_round = 999
    n_folds = 10
    early_stopping = 15
    ## GridSearch PARAMS for XGBoost
    params = {'eta': 0.02,
              'max_depth': 1,
              'subsample': 1,
              'colsample_bytree': 1,
              'objective': 'binary:logistic',
              'eval_metric':'auc',
              'min_child_weight': 1}
    ## from array format to DMatrix format :
    xg_train = xgb.DMatrix(X_train.values, label=y_train.values);
    xg_test = xgb.DMatrix(X_test, label = y_test)
    ## Baseline of 10fold CV :
    cv_results = xgb.cv(params, xg_train, num_boost_round = 999, nfold=5, early_stopping_rounds=early_stopping)



    ##############################################################
    ### GridSearch for 2 hyperparamaters :
    ### Max_Depth and Min_Child_Weight
    ###
    ##############################################################


    gridsearch_params = [
        (max_depth, min_child_weight)
        for max_depth in range(9,12)
        for min_child_weight in range(5,8)
    ]

    max_auc = 0
    best_params = None
    for max_depth, min_child_weight in gridsearch_params:
        print("CV with max_depth={}, min_child_weight={}".format(
                                 max_depth,
                                 min_child_weight))
        # Update parameters
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        # Run CV
        cv_results = xgb.cv(
            params,
            xg_train,
            num_boost_round=num_boost_round,
            nfold=5,
            metrics={'auc'},
            early_stopping_rounds=early_stopping
        )
        # Update best AUC
        mean_auc = cv_results['test-auc-mean'].max()
        boost_rounds = cv_results['test-auc-mean'].idxmax()
        print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))
        if mean_auc > max_auc:
            max_auc = mean_auc
            best_params = (max_depth,min_child_weight)
    print("Best params: {}, {}, AUC: {}".format(best_params[0], best_params[1], max_auc))
    ## update the best params :
    params['max_depth'] = best_params[0]
    params['min_child_weight'] = best_params[1]


    ##############################################################
    ### GridSearch for 2 hyperparamaters :
    ### SubSample and Col_Sample_ByTree
    ###
    ##############################################################


    gridsearch_params2 = [
        (subsample, colsample)
        for subsample in [i/10. for i in range(7,11)]
        for colsample in [i/10. for i in range(7,11)]
    ]

    max_auc = 0
    best_params = None
    for subsample, colsample in gridsearch_params2:
        print("CV with subsample={}, colsample={}".format(
                                 subsample,
                                 colsample))
        # Update parameters
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample
        # Run CV
        cv_results = xgb.cv(
            params,
            xg_train,
            num_boost_round=num_boost_round,
            nfold=5,
            metrics={'auc'},
            early_stopping_rounds=early_stopping
        )
        # Update best AUC
        mean_auc = cv_results['test-auc-mean'].max()
        boost_rounds = cv_results['test-auc-mean'].idxmax()
        print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))
        if mean_auc > max_auc:
            max_auc = mean_auc
            best_params = (subsample,colsample)
    print("Best params: {}, {}, AUC: {}".format(best_params[0], best_params[1], max_auc))
    params['subsample'] = best_params[0]
    params['colsample_bytree'] = best_params[1]

    ##############################################################
    ### GridSearch for 1 hyperparamaters :
    ### ETA : Learning Rate.
    ###
    ##############################################################

    max_auc = 0
    best_params = None
    for eta in [.3, .2, .1, .05, 0.02,.01, .005]:
        print("CV with eta={}".format(eta))
        # update our parameters
        params['eta'] = eta
        # CV
        cv_results = xgb.cv(
                params,
                xg_train,
                num_boost_round=num_boost_round,
                nfold=5,
                metrics=['auc'],
                early_stopping_rounds=early_stopping)
        # Update best score
        mean_auc = cv_results['test-auc-mean'].max()
        boost_rounds = cv_results['test-auc-mean'].idxmax()
        print("\tAUC {} for {} rounds\n".format(mean_auc, boost_rounds))
        if mean_auc > max_auc:
            max_auc = mean_auc
            best_params = eta
    print("Best params: {}, AUC: {}".format(best_params, max_auc))
    params['eta'] = best_params


    best_model = xgb.train(
        params,
        xg_train,
        num_boost_round=num_boost_round,
        evals=[(xg_test, "Test")],
        early_stopping_rounds=early_stopping)

    best_model.save_model(path + "models/xgb_model.model")
    '''
    all the code below is optional, it is for loading 
    and predicting a trained XGB model
    
    '''

    #AUC_sco = AUC_score(y_test,y_pred)

    #print('Test AUC : {}'.format(AUC_sco))
    ####
    # Load the model for future :
    ####

    # loaded_model = xgb.Booster()
    # loaded_model.load_model("MODEL_NAME.model")
    ####### And use it for predictions.
    # loaded_model.predict(dtest)

if __name__ == '__main__':
    main()