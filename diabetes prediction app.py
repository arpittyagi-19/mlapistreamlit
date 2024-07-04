# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:35:02 2024

@author: arpit
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open(r'D:\streamlit ml deployment\trained_model.sav','rb'))


#creating a function for prediction

def diabetes_prediction(input_data):
    #making a predictive system
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    

def main():
    
    #title
    st.title('Diabetes Prediction App')
    
    #getting input data from the user
    Pregnancies = st.text_input('Number of pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Diabetes pedigree value')
    Age = st.text_input('Age')
    
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,Age])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()