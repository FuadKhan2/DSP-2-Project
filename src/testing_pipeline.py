import joblib
import pandas as pd

model = joblib.load('models/log_reg_with_pipeline.pkl')

test_data = pd.read_csv('data/test.csv')

X_test = test_data.drop(columns=['Outcome'])
y_test = test_data['Outcome']

print(model.score(X_test, y_test))