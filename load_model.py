import pandas as pd
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

model = joblib.load('Keras ANN.pkl')

test_data = pd.read_csv('lung cancer survey.csv')
le = LabelEncoder()
test_data['GENDER'] = le.fit_transform(test_data['GENDER'])
test_data['LUNG_CANCER'] = le.fit_transform(test_data['LUNG_CANCER'])
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)
accuracy= accuracy_score(y_test, y_pred)

with open('accuracy.txt', 'w') as file:
    file.write(f'Accuracy: {accuracy}')