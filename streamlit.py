import streamlit as st
import joblib
import pandas as pd

#load
model = joblib.load("m1.pkl")
country_encoder = joblib.load("c1.pkl")
target_encoder = joblib.load("t1.pkl")
preprocessor = joblib.load("p1.pkl")

st.title("Purchase prediction")
country = st.selectbox("country" ,country_encoder.classes_)
age = st.number_input("Age" ,min_value = 18 , max_value = 100)
salary= st.number_input("Salary" ,min_value = 1000 , max_value = 200000)

encoded_country = country_encoder.transform([country])[0]
input_data = pd.DataFrame([[encoded_country ,age ,salary]] ,columns= ["Country" , "Age" , "Salary"])

input_scale = preprocessor.transform(input_data)

pred = model.predict(input_scale)[0]
res = target_encoder.inverse_transform([pred])[0]
btn = st.button("predict")
if btn :
    if res == "Yes" :
        st.success(res)
    else :
        st.error(res)

    


