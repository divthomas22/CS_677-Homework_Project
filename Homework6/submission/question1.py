"""
Divya Thomas 
Class: CS 677
Date: 4/24/2023
Homework Problem #1
Description of Problem (just a 1-2 line summary!): 
Since the last digit of my BUID is 2, I will split the dataset to use only class values of 1(negative) and 3(positive)
and test the accuracy of each SVM classifier in predicting the data set class values.
"""
import pandas as pd 
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Read the data into the dataframe and prepare the data for the SVM classifiers.
#function to convert the excel data into a pandas dataframe
def create_pandas_dataframe():
    #get file path
    input_dir = os.path.abspath(__file__) + '\..\datasets'
    file = os.path.join(input_dir, 'seeds_dataset.csv')

    columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'L']

    #convert to df
    full_df = pd.read_csv(file, names=columns, delimiter='\t') 

    #get a subset of the dataset containing only L = 1 and 3
    df = full_df[full_df['L'].isin([1, 3])].copy()

    # Add a class rating 1 (-) or 3(+)
    df['Class'] = np.where(df['L'] == 1, '-', '+')

    return df

df = create_pandas_dataframe()
print(df)

# Question 1.1 - Implement and analyze the accuracy of the linear kernel SVM model
print ("\n--Question 1.1--")

def split(df):
    x = df[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']]
    y = df['Class']
    x_train,x_test,y_train,y_test = train_test_split(
        x, 
        y, 
        test_size=0.5, 
        random_state=42)
    
    train = (x_train, y_train)
    test = (x_test, y_test)
    return (train, test)

split_data = split(df)

# Use linear kernel SVM classifier to predict class values for test data
def linear_kernel(train, test):

    # Fit a linear kernel SVM classifier to the training data
    lk_svm = SVC(kernel='linear')
    lk_svm.fit(train[0], train[1])

    #predict test data classes with the model
    pred = lk_svm.predict(test[0])

    #create a copy of the test data with the added predictions
    test_copy = test[0].copy()
    test_copy['Actual'] = test[1]
    test_copy['Prediction'] = pred

    return test_copy

def accuracy(dataframe):
    # get the list of actual NSP class values
    actual = dataframe['Actual'].tolist()
    # get the list of the predicted values 
    pred = dataframe["Prediction"].tolist()
    # get the total count of the predicted values 
    tot_count = len(pred)
    # set a counter for the correct predictions
    success_count = 0

    #match up each index in actual with predictions and increment success count if they match
    for i in range(0, tot_count):
        if (actual[i] == pred[i]):
            success_count += 1
    # return the percentage of success
    return (success_count / tot_count) * 100

def compute_confusion_matrix(df):
    # compute the confusion matrix
    cm = confusion_matrix(df['Actual'], df['Prediction'])
    return cm


lk = linear_kernel(split_data[0], split_data[1])
acc = accuracy(lk)
cm = compute_confusion_matrix(lk)
print(lk)
print("Accuracy: %.2f%%" % acc, "\nConfusion Matrix:\n", cm )

# Question 1.2 - Implement and analyze the accuracy of the Gaussian kernel SVM model
print ("\n--Question 1.2--")
def gaussian_kernel(train, test):

    # create a pipeline with the scaler and fit the gaussian model to the training data
    gk_svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', gamma='auto'))
    ])
    gk_svm.fit(train[0], train[1])

    #predict test data classes with the model
    pred = gk_svm.predict(test[0])

    #create a copy of the test data with the added predictions
    test_copy = test[0].copy()
    test_copy['Actual'] = test[1]
    test_copy['Prediction'] = pred

    return test_copy

gk = gaussian_kernel(split_data[0], split_data[1])
acc = accuracy(gk)
cm = compute_confusion_matrix(gk)
print(gk)
print("Accuracy: %.2f%%" % acc, "\nConfusion Matrix:\n", cm )

# Question 1.3 - Implement and analyze the accuracy of the Polynomial kernel SVM model 
print ("\n--Question 1.3--")
def poly_kernel(train, test):

    # create a pipeline with the scaler and fit the polynomial kernel DEG=3 model
    p_svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='poly', degree=3, gamma='auto', C=1))
    ])
    p_svm.fit(train[0], train[1])

    #predict test data classes with the model
    pred = p_svm.predict(test[0])

    #create a copy of the test data with the added predictions
    test_copy = test[0].copy()
    test_copy['Actual'] = test[1]
    test_copy['Prediction'] = pred

    return test_copy

pk = poly_kernel(split_data[0], split_data[1])
acc = accuracy(pk)
cm = compute_confusion_matrix(pk)
print(pk)
print("Accuracy: %.2f%%" % acc, "\nConfusion Matrix:\n", cm )