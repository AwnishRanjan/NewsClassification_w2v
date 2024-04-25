import streamlit as st
from prediction import predict



st.title("Fake News Detection")
input_text = st.text_area("Enter the news text here:")
if st.button("Predict"):
    prediction = predict(input_text)
    st.write(f"The news is: {prediction}")
