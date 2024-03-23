import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv(r'heart_disease_data.csv')
df.isnull().sum()
X = df.drop(columns='target', axis=1)
Y = df['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
model = LogisticRegression(max_iter=1000)
# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)


X_train_prediction = model.predict(X_train)

X_test_prediction = model.predict(X_test)

with open('../cardio_vascular/model_pickle', 'wb') as f:
 pickle.dump(model,f)
#
# input_data = ()#need to use pickle
#
# input_data_as_numpy_array= np.asarray(input_data)
# input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# prediction = model.predict(input_data_reshaped)
# print(prediction)
#
# if (prediction[0]== 0):
#  print('The Person does not have a Heart Disease')#need to change these
# else:
#  print('The Person has Heart Disease')#need to change these
