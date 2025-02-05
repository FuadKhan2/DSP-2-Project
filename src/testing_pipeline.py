# Import necessary libraries
import joblib
import pandas as pd

# Load the pre-trained logistic regression model with pipeline
model = joblib.load('models/log_reg_with_pipeline.pkl')

# Load the test dataset
test_data = pd.read_csv('data/test.csv')

# Separate features (X) and target variable (y)
X_test = test_data.drop(columns=['Outcome'])  # Input features
y_test = test_data['Outcome']  # True labels

# Evaluate the model on the test dataset and print the accuracy score
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy on Test Data: {accuracy:.4f}")
