import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Coding_Projects\Austin_Crime\Crime_Reports (1).csv")

family = {'Y': 1, 'N': 0}

df['Family Violence'] = df['Family Violence'].map(family)


df = df[['Family Violence','Highest Offense Description', 'Occurred Time', 'Zip Codes']]

desired_values = {'BURGLARY OF VEHICLE' : 0,
 'THEFT': 1,
 'FAMILY DISTURBANCE':2,
 'CRIMINAL MISCHIEF':3,
 'ASSAULT W/INJURY-FAM/DATE VIOL':4,
 'BURGLARY OF RESIDENCE':5,
 'DWI':6,
 'HARASSMENT':7,
 'DISTURBANCE - OTHER':8,
 'AUTO THEFT':9,
 'ASSAULT WITH INJURY':10,
 'THEFT BY SHOPLIFTING':11,
 'CUSTODY ARREST TRAFFIC WARR':12,
 'WARRANT ARREST NON TRAFFIC':13,
 'CRIMINAL TRESPASS':14,
 'BURGLARY NON RESIDENCE':15}

desired_values_list = ['BURGLARY OF VEHICLE',
 'THEFT',
 'FAMILY DISTURBANCE',
 'CRIMINAL MISCHIEF',
 'ASSAULT W/INJURY-FAM/DATE VIOL',
 'BURGLARY OF RESIDENCE',
 'DWI',
 'HARASSMENT',
 'DISTURBANCE - OTHER',
 'AUTO THEFT',
 'ASSAULT WITH INJURY',
 'THEFT BY SHOPLIFTING',
 'CUSTODY ARREST TRAFFIC WARR',
 'WARRANT ARREST NON TRAFFIC',
 'CRIMINAL TRESPASS',
 'BURGLARY NON RESIDENCE']

df = df[df['Highest Offense Description'].isin(desired_values_list)]

df['Highest Offense Description'] = df['Highest Offense Description'].map(desired_values)

df.dropna()
print(df)

X = df.drop('Highest Offense Description', axis = 'columns')
y = df['Highest Offense Description']


X = X.values
y = y.values

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16)  # Output layer])
    ])

model.compile(
    optimizer='sgd', 
    loss='mean_squared_error',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=5)

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

y_pred = model.predict(X_test)

print(y_pred)
print("Hello World")