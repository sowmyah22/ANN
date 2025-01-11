import streamlit as st
import pandas as pd
import numpy as np
import tensorflow
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from tensorflow.keras.models import load_model
import pickle as pkl



# load the model,pickle files
model=load_model('model.h5')

with open('gender_encoder.pkl','rb') as file:
     encode_gender=pkl.load(file)

with open('geo_encoder.pkl','rb') as file:
     encode_geo=pkl.load(file)

with open('stdscalr.pkl','rb') as file:
     scaler=pkl.load(file)

# Streamlit App
st.title("Customer Churn Prediction")

# UI
CreditScore=st.number_input('Credit_score')
Geography=st.selectbox('Geography',encode_geo.categories_[0])
#encode_geo.categories_[0] returns a list of arrays, and we need the first array containing the geography categories.
Gender=st.selectbox('Gender',encode_gender.classes_)
Age=st.slider('Age',18,90)
Tenure=st.slider('Tenure',0,10)
Balance=st.number_input('Balance')
NumOfProducts=st.slider('Products',1,4)
HasCrCard=st.selectbox('Has credit card',[0,1])
IsActiveMember=st.selectbox('Active Member',[0,1])
EstimatedSalary=st.number_input('Estimated Salary')

# Prepare the input data
in_data=pd.DataFrame({
     'CreditScore':[CreditScore],
     'Gender':[encode_gender.transform([Gender])[0]],
     'Age':[Age],
     'Tenure':[Tenure],
     'Balance':[Balance],
     'NumOfProducts':[NumOfProducts],
     'HasCrCard':[HasCrCard],
     'IsActiveMember':[IsActiveMember],
     'EstimatedSalary':[EstimatedSalary]
})

# Encoding of categorical variable
geo_encode=encode_geo.transform([[Geography]])
geo_encode_df=pd.DataFrame(geo_encode,columns=encode_geo.get_feature_names_out(['Geography']))

in_data=pd.concat([in_data.reset_index(drop=True),geo_encode_df],axis=1)

in_data_scaled=scaler.transform(in_data)

# prediction 
predict=model.predict(in_data_scaled)
prediction_probability=predict[0][0]

st.write(f"The preiciton probability:{prediction_probability:.2f} ")

if prediction_probability >0.5:
     st.write("The customer is likely to churn")

else:
     st.write("The Customer is not likely to churn")
