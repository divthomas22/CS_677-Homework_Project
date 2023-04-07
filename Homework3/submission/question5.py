"""
Divya Thomas 
Class: CS 677
Date: 4/1/2023
Homework Problem #5
Description of Problem (just a 1-2 line summary!):
Use the logistic regression classifier to predict test data classes and evaluate its efficiency.
"""
import question1 as q1
import question2 as q2
import question3 as q3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

bn_df = q1.create_pandas_dataframe()
train, test = q3.create_test_train(bn_df)

print ("\n--Question 5.1--")
# Question 5.1 - Logistic regression classifier used on train and test data (similar to q3.knn_classifier())
def lr_classifier(train, test):
    #create the logistic regression classifier
    lr = LogisticRegression()
    #train the logistic regression classifier
    lr.fit(train[0], train[1])

    #predict the test data class
    pred = lr.predict(test[0])

    #create another dataframe containing true values and predictions
    df = test[0].copy()
    df['color'] = test[1].copy()
    df['prediction'] = pred

    #compute the accuracy
    accuracy = q2.accuracy(df)
    return (accuracy, df)

result = lr_classifier(train, test)
print("Logistic Regression accuracy: ", result[0])

print ("\n--Question 5.2--")
# Question 5.2 - Calculate the performance metrics for the Logistic regression classifier
count_list = q2.calc_label_accuracies(result[1])
tpr = q2.calc_tpr(count_list)
tnr = q2.calc_tnr(count_list)

print('-Counts-')
print('TP: %d \nFP: %d \nTN: %d \nFN: %d \nAccuracy: %.2f \nTPR: %.2f \nTNR: %.2f' 
      % (count_list[0], count_list[3], count_list[2], count_list[1], result[0], tpr, tnr))

# Question 5.3-4 : See supplemental documenation

# Question 5.5 - Predict the class label for a bill with the last four digits of my BUID (5732)
# as feature values using the logistic regression classifier
print ("\n--Question 5.5--")
def predict_bill(f1, f2, f3, f4, train_data):

    # create a dictionary containing the bill to be tested 
    bill_data = {'f1': [f1],'f2': [f2],'f3': [f3],'f4': [f4]}
    #create dataframe from this data 
    df = pd.DataFrame(bill_data)

    #use the logistic regression classifier
    classifier = LogisticRegression()

    #train the classifier with the training set
    classifier.fit(train_data[0], train_data[1])

    #predict data using the classifier
    pred = classifier.predict(df)

    df_lr = df.copy()
    df_lr['prediction'] = pred
    print("Logistic Regression Classifier: \n", df_lr)

    return df_lr



predict_bill(5,7,3,2,train)