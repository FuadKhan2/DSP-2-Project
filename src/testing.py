# Import necessary libraries
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

# Load the test dataset
df = pd.read_csv('data/test.csv')

# Separate features (X) and target (y)
X = df.drop(columns=['Outcome', 'AgeCat', 'IgG'])
y = df['Outcome']

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include='number').columns
cat_cols = X.select_dtypes(include='object').columns

# Load the pre-trained preprocessing objects and model
num_imputer = joblib.load('models/num_imputer.pkl')   # For imputing missing numerical values
cat_imputer = joblib.load('models/cat_imputer.pkl')   # For imputing missing categorical values
scaler = joblib.load('models/scaler.pkl')             # For scaling numerical features
encoder = joblib.load('models/encoder.pkl')           # For encoding categorical features
model = joblib.load('models/log_reg.pkl')             # The pre-trained Logistic Regression model

# Apply imputers to handle missing values in the test set
X[num_cols] = num_imputer.transform(X[num_cols])
X[cat_cols] = cat_imputer.transform(X[cat_cols])

# Apply scaling to the numerical features
X[num_cols] = scaler.transform(X[num_cols])

# Apply encoding to the categorical features
X[cat_cols] = encoder.transform(X[cat_cols])

# Make predictions on the test set using the trained model
pred = model.predict(X)

# Evaluate the model by printing the accuracy score and confusion matrix
print(f"Model Accuracy: {model.score(X, y)}")
print("Confusion Matrix:")
print(confusion_matrix(y, pred))
