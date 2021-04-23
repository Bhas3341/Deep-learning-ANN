#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 19:02:16 2021

@author: bhaskaryuvaraj
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
by=pd.read_csv('/Users/bhaskaryuvaraj/Downloads/Complete-Deep-Learning-master/ANN/Churn_Modelling.csv')

#dropping columns of surname, row number and customerid since it does not provide any necessary info
by.drop(by.columns[[0,1,2]],axis=1,inplace=True)

#checking for dtypes and null values
by.dtypes #since 2 object dtypes is found, create dummy for these rows
by.isnull().sum()# no null values

#creating the dummy variables
dummy1=pd.get_dummies(by['Geography'])
dummy2=pd.get_dummies(by['Gender'])
by=pd.concat((by,dummy1,dummy2),axis=1)
by.drop(by.columns[[1,2]],axis=1,inplace=True)

#now seperating the dependent and independent variables

x=by.drop(by.columns[8],axis=1)
y=by['Exited'].copy()

#now splitting the data to train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)

#now feature scaling is done to reduce the processing time
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#now installing keras lib and packages
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

#first initialize the ANN with sequential model
ANN= Sequential()

#Now that the sequential model is created lets add the input and hidden layers and neurons

ANN.add(Dense(activation="relu", input_dim=13, units=256, kernel_initializer="he_uniform"))
ANN.add(Dense(activation="relu", units=384, kernel_initializer="he_uniform"))
ANN.add(Dense(activation="relu", units=32, kernel_initializer="he_uniform"))
ANN.add(Dense(activation="relu", units=32, kernel_initializer="he_uniform"))
ANN.add(Dense(activation="relu", units=32, kernel_initializer="he_uniform"))
ANN.add(Dense(activation="relu", units=32, kernel_initializer="he_uniform"))
ANN.add(Dense(activation="relu", units=32, kernel_initializer="he_uniform"))
ANN.add(Dense(activation="relu", units=32, kernel_initializer="he_uniform"))
ANN.add(Dense(activation="relu", units=32, kernel_initializer="he_uniform"))
ANN.add(Dense(activation="relu", units=32, kernel_initializer="he_uniform"))
#output layer
ANN.add(Dense(activation="sigmoid",  units=1, kernel_initializer="glorot_uniform"))

#now after creating the model compile it

ANN.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

#now fitting the data to ANN model
model=ANN.fit(x_train,y_train,validation_split=0.33,batch_size=10, epochs=100)


#Now predicting the model
y_pred=ANN.predict(x_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)

#now lets do hyper parameter testing to find the best number of hidden layers and neurons and also the learning rate


