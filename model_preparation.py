print(f'Hello from {__file__}')

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import os

X_train = pd.read_csv('./data/processed/X_train.csv', index_col=0)
y_train = pd.read_csv('./data/processed/y_train.csv', index_col=0)
X_test = pd.read_csv('./data/processed/X_test.csv', index_col=0)
y_test = pd.read_csv('./data/processed/y_test.csv', index_col=0)


model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, 
                                max_depth=10, max_features=10, 
                                random_state=42)
model.fit(X_train, y_train)

model_score_train = model.score(X_train, y_train)
model_score_test = model.score(X_test, y_test)
print(f"Правильность на обучающем наборе: {model.score(X_train, y_train)}")
print(f"Правильность на тестовом наборе: {model.score(X_test, y_test)}")

path = './model/'
if not os.path.exists(path):
    os.mkdir(path)

filename = os.path.join(path, 'model.pkl')
pickle.dump(model, open(filename, 'wb'))
