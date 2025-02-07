# Import necessary libraries
import joblib
import pandas as pd

# Define a sample data input (representing a new individual for prediction)
sample_data = {
    'Gender': 'Male',       # Gender of the individual
    'Age': 24,              # Age of the individual
    'NS1': 0,               # NS1 feature value (assumed)
    'IgG': 0,               # IgG feature value (assumed)
    'IgM': 0,               # IgM feature value (assumed)
    'Area': 'Mirpur',       # Area of residence
    'AreaType': 'Undeveloped',  # Type of area (e.g., developed, undeveloped)
    'HouseType': 'Building',    # Type of house (e.g., apartment, house)
    'District': 'Dhaka'         # District of residence
}

# Convert the sample data into a DataFrame (suitable for input to the model)
sample_data_df = pd.DataFrame([sample_data])

# Load the pre-trained model pipeline (which includes both preprocessing and logistic regression model)
model = joblib.load('models/log_reg_with_pipeline.pkl')

# Make a prediction using the model and sample data
result = model.predict(sample_data_df)

# Print the prediction result (whether the outcome is positive or negative)
print(result)
