import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss, f1_score, make_scorer

import numpy as np
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
import string
from make_submission import make_submission


def create_features(data):
    for col in ['bill_amt5', 'bill_amt6', 'pay_amt5', 'pay_amt6', 'pay_5', 'pay_6']:
        data[col] = data[col].fillna(0)
    data['bill_amt_avg'] = data[['bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5',
                                 'bill_amt6']].mean(axis=1)
    data['pay_amt_avg'] = data[['pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5',
                                'pay_amt6']].mean(axis=1)
    for col in range(1, 7):
        data['diff_amt' + str(col)] = data['pay_amt' + str(col)] - data['bill_amt' + str(col)]

    education_limit_summary = data.groupby("education").agg({"limit_bal": [np.mean, np.max, np.min],
                                                             "bill_amt_avg": [np.mean, np.max, np.min],
                                                             "pay_amt_avg": [np.mean, np.max, np.min]})
    education_limit_summary.columns = education_limit_summary.columns.map('_'.join).to_series().map(
        lambda x: "education_" + x)
    education_limit_summary = education_limit_summary.reset_index()

    data['age_bins'] = pd.qcut(data['age'], 6, labels=[
        "age_lt_26", "age_bt_26_30", "age_bt_30_34",
        "age_bt_34_39", "age_bt_39_45", "age_gt_45"])  # 6 decided by analysis in the next section woe_iv
    age_limit_summary = data.groupby("age_bins").agg({"limit_bal": [np.mean, np.max, np.min],
                                                      "bill_amt_avg": [np.mean, np.max, np.min],
                                                      "pay_amt_avg": [np.mean, np.max, np.min]})
    age_limit_summary.columns = age_limit_summary.columns.map('_'.join).to_series().map(
        lambda x: "age_" + x)
    age_limit_summary = age_limit_summary.reset_index()

    marriage_limit_summary = data.groupby("marriage").agg({"limit_bal": [np.mean, np.max, np.min],
                                                           "bill_amt_avg": [np.mean, np.max, np.min],
                                                           "pay_amt_avg": [np.mean, np.max, np.min]})
    marriage_limit_summary.columns = marriage_limit_summary.columns.map('_'.join).to_series().map(
        lambda x: "marriage_" + x)
    marriage_limit_summary = marriage_limit_summary.reset_index()

    data = data \
        .merge(education_limit_summary, on='education') \
        .merge(age_limit_summary, on="age_bins") \
        .merge(marriage_limit_summary, on="marriage")
    return data


def preprocess_data(data, numerical_columns, target):
    data[numerical_columns] = RobustScaler().fit_transform(data[numerical_columns])
    return data


def create_model(data, numerical_columns, categorical_columns, target):
    mutual = SelectKBest(score_func=mutual_info_classif, k=40)
    X_train, X_test, y_train, y_test = train_test_split(data[numerical_columns + categorical_columns],
                                                        data[target], test_size=0.25, random_state=42,
                                                        stratify=data[target])
    mutual.fit(X_train, y_train)
    X_test_mut = mutual.transform(X_test)
    cols = mutual.get_support(indices=True)
    selected_columns = X_train.iloc[:, cols].columns.tolist()

    clf = GradientBoostingClassifier()
    clf.fit(data[selected_columns], data[target])
    return clf, selected_columns

def evaluate_model(model, data, selected_columns, target):
    scores = cross_val_score(model, data[selected_columns], data[target], cv=5, scoring='f1_macro')
    print("F1 scores CV: ", scores)

    logLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
    scores = cross_val_score(model, data[selected_columns], data[target], cv=5, scoring=logLoss)
    print("Log loss CV: ", scores)

def main():
    data = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    numerical_columns = ['limit_bal', 'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
                         'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6',
                         'bill_amt_avg', 'pay_amt_avg', 'diff_amt1', 'diff_amt2', 'diff_amt3',
                         'diff_amt4', 'diff_amt5', 'diff_amt6',
                         'education_limit_bal_mean', 'education_limit_bal_amax',
                         'education_limit_bal_amin', 'education_bill_amt_avg_mean',
                         'education_bill_amt_avg_amax', 'education_bill_amt_avg_amin',
                         'education_pay_amt_avg_mean', 'education_pay_amt_avg_amax',
                         'education_pay_amt_avg_amin', 'age_limit_bal_mean',
                         'age_limit_bal_amax', 'age_limit_bal_amin', 'age_bill_amt_avg_mean',
                         'age_bill_amt_avg_amax', 'age_bill_amt_avg_amin',
                         'age_pay_amt_avg_mean', 'age_pay_amt_avg_amax', 'age_pay_amt_avg_amin',
                         'marriage_limit_bal_mean', 'marriage_limit_bal_amax',
                         'marriage_limit_bal_amin', 'marriage_bill_amt_avg_mean',
                         'marriage_bill_amt_avg_amax', 'marriage_bill_amt_avg_amin',
                         'marriage_pay_amt_avg_mean', 'marriage_pay_amt_avg_amax',
                         'marriage_pay_amt_avg_amin'
                         ]
    categorical_columns = ['pay_1', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6', 'education']
    target = 'default_oct'

    data = create_features(data)
    data = preprocess_data(data, numerical_columns, target)
    data[target] = data[target].map({'yes': 1, 'no': 0})
    model, selected_columns = create_model(data, numerical_columns, categorical_columns, target)
    evaluate_model(model, data, selected_columns, target)

    test = create_features(test)
    test = preprocess_data(test, numerical_columns, target)
    test['pr_y'] = model.predict_proba(test[selected_columns])[:, 1]
    response = make_submission("shashank_bassi", test[["customer_id", "pr_y"]])
    print(response)

if __name__ == "__main__":
    main()
