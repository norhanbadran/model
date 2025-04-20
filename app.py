import streamlit as st 

st.write("welcome")
email_true = 'ahmedsamy@gmail.com'

email = st.text_input("enter your email:")

if email_true == email :
    password = st.text_input("enter your password:")
    
else:
    st.error("invalid email please try again.")