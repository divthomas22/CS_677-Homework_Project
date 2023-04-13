"""
Divya Thomas 
Class: CS 677
Date: 4/8/2023
Homework Problem #2
Description of Problem (just a 1-2 line summary!):
Perform predictions using the five models and calculate the estimated loss for each. 
"""
import os
import question1 as q1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df_0, df_1 = q1.create_pandas_dataframes()

# Question 2 - Split each df up into train and test data (X: serum sodium, Y : serum creatinine)
def split(df):
    x_train,x_test,y_train,y_test = train_test_split(
        df[['serum_sodium']], 
        df['serum_creatinine'], 
        test_size=0.5, 
        random_state=0)
    
    train = (x_train, y_train)
    test = (x_test, y_test)
    return (train, test)

#function to plot predicted values as a line graph and actual as a scatter plot
def create_plot(test, y_pred, name):
    #plot the predicted values as a line graph
    fig,ax = plt.subplots()
    ax.set_xlabel('serum_sodium')
    ax.set_ylabel('serum_creatinine')
    ax.plot(test[0], y_pred, color='blue', label='Predicted')

    # plot the actual values as a scatter plot
    ax.scatter(test[0], test[1], color='red', label='Actual')

    ax.legend()

    #file location to save the plot
    input_dir = os.path.abspath(__file__) + '\..\datasets\q2'
    plot_file = os.path.join(input_dir, name)

    plt.savefig(plot_file)

    print("Plot saved to " + str(plot_file))

# function to compute SSE (L = sum((y_actual-y_pred)**2)) 
def compute_sse(actual, pred):
    # create a temp df containing actual and predicted values
    df = pd.DataFrame()
    df['actual'] = actual
    df['predicted'] = pred

    # L - SSE sum of the squared of residuals
    sum = 0
    # Iterate through each row of the DataFrame
    for index, row in df.iterrows():
        # get each y value 
        y_actual = row['actual']
        y_pred = row['predicted']

        # compute the residual and square it 
        res = y_actual - y_pred
        res_sq = (res)**2

        # add it to the sum 
        sum += res_sq

    sum = np.round(sum, 3)
    
    print ("SSE = " + str(sum))

train0, test0 = split(df_0)
train1, test1 = split(df_1)

print ("\n--Question 2.1 (Linear Regression)--")
# Question 2.1 - Use linear regression to predict the test data values
def predict_lr(train, test, name):
    lr = LinearRegression()

    # a. fit the model
    lr.fit(train[0], train[1])

    # b. Print the weights a and b 
    print ("Weights: a=%.3f , b=%.3f" % (lr.coef_[0], lr.intercept_))

    # c. predict test data values and round to 2 decimal places
    pred = np.round(lr.predict(test[0]),2)

    # d. plot actual and predicted values and save to file
    create_plot(test, pred, name)

    # e. compute the loss function (SSE)
    compute_sse(test[1], pred)


print("\nSurviving Patients:")
predict_lr(train0, test0, "LR_0.png")

print("\nDeceased Patients:")
predict_lr(train1, test1, "LR_1.png")


print ("\n--Question 2.2 (Quadratic)--")
# Question 2.2 - Use quadratic regression to predict the test data values
def predict_q(train, test, name):
    q = PolynomialFeatures(degree=2, include_bias=False)

    # Transform x data to polynomial features
    x_train = q.fit_transform(train[0])
    x_test = q.transform(test[0])

    # a. fit the model
    lr = LinearRegression()
    lr.fit(x_train, train[1])

    # b. Print the weights a, b and c
    print ("Weights: a=%.3f , b=%.3f, c=%.3f" % (lr.coef_[0], lr.coef_[1], lr.intercept_))

    # c. predict test data values and round to 2 decimal places
    pred = np.round(lr.predict(x_test),2)

    # d. plot actual and predicted values and save to file
    create_plot(test, pred, name)

    # e. compute the loss function (SSE)
    compute_sse(test[1], pred)

print("\nSurviving Patients:")
predict_q(train0, test0, "Q_0.png")

print("\nDeceased Patients:")
predict_q(train1, test1, "Q_1.png")

print ("\n--Question 2.3 (Cubic Spline)--")
# Question 2.3 - Use Cubic Spline model to predict the test data values
def predict_cs(train, test, name):
    cs = PolynomialFeatures(degree=3, include_bias=False)

    # Transform x data to polynomial features
    x_train = cs.fit_transform(train[0])
    x_test = cs.transform(test[0])

    # a. fit the model
    lr = LinearRegression()
    lr.fit(x_train, train[1])

    # b. Print the weights a, b, c, and d
    print ("Weights: a=%.3f , b=%.3f, c=%.3f, d=%.3f" % (lr.coef_[0], lr.coef_[1], lr.coef_[2], lr.intercept_))

    # c. predict test data values and round to 2 decimal places
    pred = np.round(lr.predict(x_test),2)

    # d. plot actual and predicted values and save to file
    create_plot(test, pred, name)

    # e. compute the loss function (SSE)
    compute_sse(test[1], pred)

print("\nSurviving Patients:")
predict_cs(train0, test0, "CS_0.png")

print("\nDeceased Patients:")
predict_cs(train1, test1, "CS_1.png")


print ("\n--Question 2.4 (GLM1)--")
# Question 2.4 - Use first GLM model to predict the test data values
def predict_glm1(train, test, name):

    glm = PolynomialFeatures(degree=1, include_bias=False)

    # Transform x data to polynomial features
    x_train = glm.fit_transform(np.log(train[0]))
    x_test = glm.transform(np.log(test[0]))

    # a. fit the model
    lr = LinearRegression()
    lr.fit(x_train, train[1])

    # b. Print the weights a, b
    print ("Weights: a=%.3f , b=%.3f" % (lr.coef_[0], lr.intercept_))

    # c. predict test data values and round to 2 decimal places
    pred = np.round(lr.predict(x_test),2)

    # d. plot actual and predicted values and save to file
    create_plot(test, pred, name)

    # e. compute the loss function (SSE)
    compute_sse(test[1], pred)

print("\nSurviving Patients:")
predict_glm1(train0, test0, "GLM1_0.png")

print("\nDeceased Patients:")
predict_glm1(train1, test1, "GLM1_1.png")


print ("\n--Question 2.5 (GLM2)--")
# Question 2.5 - Use second GLM model to predict the test data values
def predict_glm2(train, test, name):

    glm = PolynomialFeatures(degree=1, include_bias=False)

    # Transform x data to polynomial features
    x_train = glm.fit_transform(np.log(train[0]))
    x_test = glm.transform(np.log(test[0]))

    # a. fit the model (apply log to y as well)
    lr = LinearRegression()
    lr.fit(x_train, np.log(train[1]))

    # b. Print the weights a, b
    print ("Weights: a=%.3f , b=%.3f" % (lr.coef_[0], lr.intercept_))

    # c. predict test data values and round to 2 decimal places
    y_log_pred = lr.predict(x_test)

    # Convert the predicted values from logy to y and round to 2 decimal places
    pred = np.round(np.exp(y_log_pred),2)

    # d. plot actual and predicted values and save to file
    create_plot(test, pred, name)

    # e. compute the loss function (SSE)
    compute_sse(test[1], pred)

print("\nSurviving Patients:")
predict_glm2(train0, test0, "GLM2_0.png")

print("\nDeceased Patients:")
predict_glm2(train1, test1, "GLM2_1.png")