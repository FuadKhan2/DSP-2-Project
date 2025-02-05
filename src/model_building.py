# Import necessary libraries
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# Load the dataset
data = pd.read_csv('data/dengue.csv')
print(data.head())

# Create 'AgeCat' feature by categorizing 'Age' into different groups
data['AgeCat'] = pd.cut(data['Age'], bins=[-np.inf, 18, 30, 45, np.inf], labels=['child', 'young', 'middle-aged', 'aged'])

# Split the data into training and test sets (80% training, 20% testing)
train, test = train_test_split(data, test_size=0.2, random_state=7, stratify=data['AgeCat'])

# Create a directory for data if it doesn't exist
os.makedirs('data', exist_ok=True)

# Save the train and test datasets as CSV files
train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)

# Further split the training set into training and validation sets (80% train, 20% validation)
train_set, val_set = train_test_split(train, test_size=0.2, random_state=7, stratify=train['AgeCat'])

# Drop the 'AgeCat' column from the training and validation sets as it is no longer needed
train_set.drop(columns=['AgeCat'], axis=1, inplace=True)
val_set.drop(columns=['AgeCat'], axis=1, inplace=True)

# Define features (X) and target (y) for both training and validation sets
X_train = train_set.drop(columns=['Outcome', 'IgG'])
y_train = train_set['Outcome']
X_val = val_set.drop(columns=['Outcome', 'IgG'])
y_val = val_set['Outcome']

# Identify numerical and categorical columns
num_cols = X_train.select_dtypes(include='number').columns
cat_cols = X_train.select_dtypes(include='object').columns

# Initialize imputers to handle missing data
num_imputer = SimpleImputer(strategy='mean')  # For numerical columns
cat_imputer = SimpleImputer(strategy='most_frequent')  # For categorical columns

# Impute missing values for numerical and categorical columns in both train and validation sets
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])

X_val[num_cols] = num_imputer.transform(X_val[num_cols])
X_val[cat_cols] = cat_imputer.transform(X_val[cat_cols])

# Compute Q1 (25th percentile) and Q3 (75th percentile) for outlier detection in 'Age' feature
Q1 = data['Age'].quantile(0.25)
Q3 = data['Age'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds for detecting outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detect and print outliers in the 'Age' feature
outliers = data[(data['Age'] < lower_bound) | (data['Age'] > upper_bound)]
print(outliers)

# Initialize scalers and encoders for data preprocessing
scaler = StandardScaler()  # For scaling numerical features
encoder = OrdinalEncoder()  # For encoding categorical features

# Apply scaling to numerical features in both train and validation sets
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])

# Apply ordinal encoding to categorical features in both train and validation sets
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_val[cat_cols] = encoder.transform(X_val[cat_cols])

# Train and evaluate multiple models to compare their performance

# Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_val)
print(f"Logistic Regression Accuracy: {log_reg.score(X_val, y_val)}")
print(confusion_matrix(y_val, log_reg_pred))

# K-Nearest Neighbors (KNN) Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_val)
print(f"KNN Accuracy: {knn.score(X_val, y_val)}")
print(confusion_matrix(y_val, knn_pred))

# Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=7)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
print(f"Random Forest Accuracy: {rf.score(X_val, y_val)}")
print(confusion_matrix(y_val, rf_pred))

# Decision Tree Model
dt = DecisionTreeClassifier(random_state=7)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_val)
print(f"Decision Tree Accuracy: {dt.score(X_val, y_val)}")
print(confusion_matrix(y_val, dt_pred))

# Final Model Selection: Logistic Regression
# Based on the accuracy, the Logistic Regression model is selected as the final model

# Save the final model and preprocessing objects
os.makedirs('models', exist_ok=True)

joblib.dump(log_reg, 'models/log_reg.pkl')  # Logistic Regression model
joblib.dump(scaler, 'models/scaler.pkl')    # Scaler for numerical features
joblib.dump(encoder, 'models/encoder.pkl')  # Encoder for categorical features
joblib.dump(num_imputer, 'models/num_imputer.pkl')  # Imputer for numerical features
joblib.dump(cat_imputer, 'models/cat_imputer.pkl')  # Imputer for categorical features

