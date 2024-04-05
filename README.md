# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Import the necessary packages & modules

### STEP2:

Load and read the dataset.

### STEP 3:

Perform pre processing and clean the dataset.

### STEP 4:

Encode categorical value into numerical values using ordinal/label/one hot encoding.

### STEP 5:

Visualize the data using different plots in seaborn.


## PROGRAM

### Name: Nivetha A
### Register Number:212222230101

```
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt

customer_df = pd.read_csv('/content/customers (1).csv')

customer_df.columns

customer_df.dtypes

customer_df.shape

customer_df.isnull().sum()

customer_df_cleaned = customer_df.dropna(axis=0)

customer_df_cleaned.isnull().sum()

customer_df_cleaned.shape

customer_df_cleaned.dtypes

customer_df_cleaned['Gender'].unique()

customer_df_cleaned['Ever_Married'].unique()

customer_df_cleaned['Graduated'].unique()

customer_df_cleaned['Profession'].unique()

customer_df_cleaned['Spending_Score'].unique()

customer_df_cleaned['Var_1'].unique()

customer_df_cleaned['Segmentation'].unique()

categories_list=[['Male', 'Female'],
           ['No', 'Yes'],
           ['No', 'Yes'],
           ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
            'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
           ['Low', 'Average', 'High']
           ]
enc = OrdinalEncoder(categories=categories_list)

customers_1 = customer_df_cleaned.copy()

customers_1[['Gender',
             'Ever_Married',
              'Graduated','Profession',
              'Spending_Score']] = enc.fit_transform(customers_1[['Gender',
                                                                 'Ever_Married',
                                                                 'Graduated','Profession',
                                                                 'Spending_Score']])

customers_1.dtypes

le = LabelEncoder()

customers_1['Segmentation'] = le.fit_transform(customers_1['Segmentation'])

customers_1.dtypes

customers_1 = customers_1.drop('ID',axis=1)
customers_1 = customers_1.drop('Var_1',axis=1)

customers_1.dtypes

corr = customers_1.corr()

sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)

sns.pairplot(customers_1)

sns.distplot(customers_1['Age'])

plt.figure(figsize=(10,6))
sns.countplot(customers_1['Family_Size'])

plt.figure(figsize=(10,6))
sns.boxplot(x='Family_Size',y='Age',data=customers_1)

plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Spending_Score',data=customers_1)

plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Age',data=customers_1)

customers_1.describe()

customers_1['Segmentation'].unique()

X=customers_1[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size']].values

y1 = customers_1[['Segmentation']].values

one_hot_enc = OneHotEncoder()

one_hot_enc.fit(y1)

y1.shape

y = one_hot_enc.transform(y1).toarray()

y.shape

y1[0]

y[0]

X.shape

X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.33,
                                               random_state=50)

X_train[0]

X_train.shape

scaler_age = MinMaxScaler()

scaler_age.fit(X_train[:,2].reshape(-1,1))

X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)

X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)

AI=Sequential([
    Dense(units=5,activation='relu',input_shape=[8]),
    Dense(units=3,activation='relu'),
    Dense(units=4,activation='relu'),
    Dense(units=4,activation='softmax')
])

AI.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=2)

metrics = pd.DataFrame(AI.history.history)

metrics.head()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(AI.predict(X_test_scaled), axis=1)

x_test_predictions.shape

y_test_truevalue = np.argmax(y_test,axis=1)

y_test_truevalue.shape

print(confusion_matrix(y_test_truevalue,x_test_predictions))

print(classification_report(y_test_truevalue,x_test_predictions))

AI.save('customer_classification_model.h5')

AI = load_model('customer_classification_model.h5')

ai_brain = load_model('customer_classification_model.h5')

x_single_prediction = np.argmax(ai_brain.predict(X_test_scaled[1:2,:]), axis=1)

print(x_single_prediction)

print(le.inverse_transform(x_single_prediction))

```

## Dataset Information

![315347348-da665364-c61a-44b7-be9b-42884a75b24e](https://github.com/nivetharajaa/nn-classification/assets/120543388/cdfe3e8c-d3de-45ba-a83a-c318bfbea78f)


## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot

![315348360-3e2928e0-641e-43df-ba7a-7bf1e009613c](https://github.com/nivetharajaa/nn-classification/assets/120543388/b411578c-18e5-4899-8d13-651b2867f691)


### Classification Report

![315348987-8013bbce-6540-4982-bea3-74d82f1b2618](https://github.com/nivetharajaa/nn-classification/assets/120543388/8b11b90e-aad3-4201-becf-0c6cce6b3ea7)


### Confusion Matrix

![315350342-b2a03f08-13f5-4038-a5da-b86bc3dd1191](https://github.com/nivetharajaa/nn-classification/assets/120543388/651b800c-dec3-4538-bdf0-1440f18c057d)

### New Sample Data Prediction

![315351887-0d65379a-0b22-4294-ba05-97bd2d5309b5](https://github.com/nivetharajaa/nn-classification/assets/120543388/36d0797e-bab1-4ed6-997f-c3cb83b48c03)


## RESULT
A neural network classification model is developed for the given dataset.
