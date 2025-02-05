import streamlit as st
import pandas as pd
import joblib

st.title('Dengue Prediction App')
st.sidebar.header('User Input Parameters')

Gender = st.sidebar.selectbox('Gender', ['Male', 'Female']) 
Age = st.sidebar.slider('Age', min_value=0, max_value=100)     
NS1 = st.sidebar.selectbox('NS1', [0, 1])
IgG = st.sidebar.selectbox('IgG', [0, 1])
IgM = st.sidebar.selectbox('IgM', [0, 1])
Area = st.sidebar.selectbox('Area', ['Mirpur', 'Chawkbazar', 'Paltan', 'Motijheel', 'Gendaria',
       'Dhanmondi', 'New Market', 'Sher-e-Bangla Nagar', 'Kafrul',
       'Pallabi', 'Mohammadpur', 'Shahbagh', 'Shyampur', 'Kalabagan',
       'Bosila', 'Jatrabari', 'Adabor', 'Kamrangirchar', 'Biman Bandar',
       'Ramna', 'Badda', 'Bangshal', 'Sabujbagh', 'Hazaribagh',
       'Sutrapur', 'Lalbagh', 'Demra', 'Banasree', 'Cantonment',
       'Keraniganj', 'Tejgaon', 'Khilkhet', 'Kadamtali', 'Gulshan',
       'Rampura', 'Khilgaon'])       
AreaType = st.sidebar.selectbox('AreaType', ['Undeveloped', 'Developed'])  
HouseType = st.sidebar.selectbox('HouseType', ['Building', 'Other', 'Tinshed'])
District = st.sidebar.selectbox('District', ['Dhaka'])

input_data = {
    'Gender': Gender,    
    'Age': Age,       
    'NS1': NS1,        
    'IgG': IgG,         
    'IgM': IgM,        
    'Area': Area,        
    'AreaType': AreaType,   
    'HouseType': HouseType,    
    'District': District         
}

input_data_df = pd.DataFrame([input_data])

model = joblib.load('G:\Dengue\models\log_reg_with_pipeline.pkl')
result = model.predict(input_data_df)

st.table(input_data_df)

txt = ''
if result == 0:
    txt = 'negative'
else:
    txt = 'positive'

st.metric('Dengue Prediction', f'You Are Dengue {txt}.')