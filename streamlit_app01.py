#Importing Libraries
import streamlit as st 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
#pickle Importing
import joblib

#Loading the  pre-trained model

model = joblib.load('Keras ANN.pkl')

#Loading Display and Accuracy
with open('accuracy.txt', 'r') as file:
    accuracy = file.read()

st.title("Model Accuracy and Lungs Cancer Prediction")
st.write(f"model {accuracy}")

#User Inputs for real-time predictions
st.header("Real-Time Prediction")


#Loading the test data
test_data = pd.read_csv('lung cancer survey.csv')
le = LabelEncoder()
test_data['GENDER'] = le.fit_transform(test_data['GENDER'])
test_data['LUNG_CANCER'] = le.fit_transform(test_data['LUNG_CANCER'])
#Assuming the last column is test
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

#Assuming the model expects the same input features as X_test
input_data = []
for col in X_test.columns:
    input_value = st.number_input(f"input for {col}", value=0.0)
    input_data.append(input_value)

#Dataframe for prediction
input_df = pd.DataFrame([input_data], columns=X_test.columns)

#Make Predictions
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f"Prediction: {prediction[0]}")

#Ploting Summary
st.header("Accuracy Plot")
st.bar_chart([float(accuracy.split(':' )[1])])