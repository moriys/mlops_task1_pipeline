print(f'Hello from {__file__}')

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

X = pd.read_csv('./data/raw/train/Train.csv', delimiter=',')
y = pd.read_csv('./data/raw/train/Target.csv', delimiter=',')
X_kaggle = pd.read_csv('./data/raw/test/Test.csv', delimiter=',')
X_full = pd.concat([X, X_kaggle], sort=False, axis=0)

cat_columns = ['code', 'period', 'id', 'Country']
num_columns = ['year', 'tourists', 'venue', 'rate', 'food',
               'glass', 'metal', 'other', 'paper', 'plastic',
               'leather', 'green_waste', 'waste_recycling']

X_full_num = X_full[num_columns].copy()
X_full_cat = X_full[cat_columns].copy()

scaler = MinMaxScaler().fit(X_full_num)
X_full_num_tr = scaler.transform(X_full_num)
X_full_num_tr = pd.DataFrame(X_full_num_tr, columns=num_columns)

onehot = OneHotEncoder().fit(X_full_cat)
X_full_cat_tr = onehot.transform(X_full_cat).toarray()
X_full_cat_tr = pd.DataFrame(X_full_cat_tr)

X_full_tr = pd.concat([X_full_num_tr, X_full_cat_tr], sort=False, axis=1)

X_tr = X_full_tr.iloc[:X.shape[0],:]
X_kaggle_tr = X_full_tr.iloc[X.shape[0]:,:]


X_train, X_test, y_train, y_test = train_test_split(
    X_tr, y['polution_clf'], test_size=0.01, random_state=42)


path = './data/processed/'
if not os.path.exists(path):
    os.mkdir(path)

X_train.to_csv('./data/processed/X_train.csv')
X_test.to_csv('./data/processed/X_test.csv')
y_train.to_csv('./data/processed/y_train.csv')
y_test.to_csv('./data/processed/y_test.csv')
X_kaggle_tr.to_csv('./data/processed/X_for_kaggle.csv')
