import joblib
import pandas as pd

sample_data = {
    'Gender': 'Male',    
    'Age': 24,       
    'NS1': 0,        
    'IgG': 0,         
    'IgM': 0,        
    'Area': 'Mirpur',        
    'AreaType': 'Undeveloped',   
    'HouseType': 'Building',    
    'District': 'Dhaka'         
}

sample_data_df = pd.DataFrame([sample_data])

model = joblib.load('models/log_reg_with_pipeline.pkl')
result = model.predict(sample_data_df)
print(result)