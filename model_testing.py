print(f'Hello from {__file__}')

import os
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier


X_kaggle = pd.read_csv('./data/raw/test/Test.csv', delimiter=',')
X_kaggle_tr = pd.read_csv('./data/processed/X_for_kaggle.csv', delimiter=',', index_col=0)

X_kaggle_idx = X_kaggle['Unnamed: 0'].copy()
X_kaggle_idx.rename('idx', inplace=True)

model = pickle.load(open('./model/model.pkl', 'rb'))

y_predict = model.predict(X_kaggle_tr)
y_predict = pd.DataFrame(y_predict, columns=['polution_clf',])

y_predict = pd.concat([X_kaggle_idx, y_predict], sort=False, axis=1)

path = './result/'
if not os.path.exists(path):
    os.mkdir(path)

y_predict.to_csv('./result/Submission.csv', index = False)
print(f"Результат загружен в файл 'Submission.csv' в папке : {path}")
