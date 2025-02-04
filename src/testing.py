import numpy as np
import pandas as pd

df = pd.read_csv('data/test.csv')
X = df.drop(columns=['Outcome', 'AgeCat'])
y = df['Outcome']

num_cols = X.select_dtypes(include='number').columns
cat_cols = X.select_dtypes(include='object').columns

import joblib

num_imputer = joblib.load('models/num_imputer.pkl')
cat_imputer = joblib.load('models/cat_imputer.pkl')
scaler = joblib.load('models/scaler.pkl')
encoder = joblib.load('models/encoder.pkl')
model = joblib.load('models/log_reg.pkl')

X[num_cols] = num_imputer.transform(X[num_cols])
X[cat_cols] = cat_imputer.transform(X[cat_cols])

X[num_cols] = scaler.transform(X[num_cols])
X[cat_cols] = encoder.transform(X[cat_cols])

pred = model.predict(X)
print(model.score(X, y))

