import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from typing import Union
import os
from sklearn.preprocessing import RobustScaler


class PreprocessEstimator(BaseEstimator, TransformerMixin):
    def __init__(self,
                 categorical_variables,
                 continuous_variables
                 ):
        self.categorical_variables = categorical_variables
        self.continuous_variables = continuous_variables
        self.continuous_scaler = RobustScaler()
        self.new_feats = []

    def transform_categorical(self, X: pd.DataFrame):
        for col in self.categorical_variables:
            dummy = pd.get_dummies(X[col], prefix=col)
            new_fts = list(dummy.columns)
            self.new_feats.extend(new_fts)
            X = X.join(dummy)
        return X

    def fit(self, X: pd.DataFrame, y=None):
        self.continuous_scaler = self.continuous_scaler.fit(X[self.continuous_variables])
        return self

    def transform(self, X):
        X = self.transform_categorical(X)
        X[self.continuous_variables] = self.continuous_scaler.transform(X[self.continuous_variables])
        return X


class ModelBuilder(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 categorical_variables=[],
                 continuous_variables=[],
                 ordinal_variables=[]
                 ):

        self.preprocessor = PreprocessEstimator(categorical_variables, continuous_variables)
        self.categorical_variables = categorical_variables
        self.continuous_variables = continuous_variables
        self.ordinal_variables = ordinal_variables

    @property
    def cols_toberemoved(self):
        return ['gender_Unknown', 'prob_has_diabetes', 'avg_years_educ', 'adi_natrank', 'email_TEXTONLY',
                'email_PROGRAMOVERVIEW']

    def fit(self, X, y):

        X = self.preprocessor.fit_transform(X)
        self.prediction_columns = self.preprocessor.new_feats + \
                                  self.continuous_variables + \
                                  self.ordinal_variables
        for i in self.cols_toberemoved:
            try:
                self.prediction_columns.remove(i)
            except:
                continue
        self.model = RandomForestClassifier().fit(X[self.prediction_columns], y)

    def predict(self, X):
        X = self.preprocessor.transform(X)
        return self.model.predict(X[self.prediction_columns])


def runner(
        data_path='./data/data.parquet',
        data_type='parquet',
):
    """

    :param data_path: location of dataset.
    :param data_type: type of data file, parquet or csv for now.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError("Incorrect data path, file doesn't exist")
    if data_type == 'parquet':
        data = pd.read_parquet(data_path)
    elif data_type == 'csv':
        data = pd.read_csv(data_path)
    dtypes = defaultdict(list)

    for i in data.columns:
        if i != 'user_uuid':
            dtypes[str(data[i].dtype)].append(i)

    categorical_variables = dtypes['object']
    continuous_variables = dtypes['float64']
    ordinal_variables = dtypes['int64']

    data = data[data[
                    categorical_variables + \
                    continuous_variables + \
                    ordinal_variables
                    ].isnull().sum(axis=1) == 0]

    print('Dropped few rows, remaining number of rows=', data.shape[0])

    data['clicked_email'] = data['first_click_time_utc'].isnull() == False
    X_train, X_test, y_train, y_test = train_test_split(data, data['clicked_email'],
                                                        test_size=0.2, random_state=42,
                                                        stratify=data['clicked_email'])

    model = ModelBuilder(
        categorical_variables=dtypes['object'],
        continuous_variables=dtypes['float64'],
        ordinal_variables=dtypes['int64']
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


if __name__=='__main__':
    runner()