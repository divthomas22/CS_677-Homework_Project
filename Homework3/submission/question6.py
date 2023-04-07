"""
Divya Thomas 
Class: CS 677
Date: 4/1/2023
Homework Problem #6
Description of Problem (just a 1-2 line summary!):
Using feature selection techniques, determine the effect of each feature on the accuracy of the 
logistic regression classifier.
"""
import question5 as q5
import question1 as q1
from sklearn.model_selection import train_test_split

bn_df = q1.create_pandas_dataframe()

print ("\n--Question 6.1--")
# Question 4.1 - Calculate the accuracy of the logistic regression classifier with one feature missing
def calc_lr_without_feature(dataframe, f):
       x = dataframe[['f1', 'f2', 'f3', 'f4']]
       y = dataframe['color']

       #remove the column specified if it exists
       if ('f'+str(f) in dataframe.columns.tolist()):
              x.pop('f'+str(f))
       
       #split dataset 50/50 into a training and testing set 
       x_train,x_test,y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

       train = (x_train, y_train)
       test = (x_test, y_test)

       result = q5.lr_classifier(train, test) 
       # will return the accuracy value and prediction-containing dataframe as a tuple

       return result[0] #just return the accuracy

def calc_f_missing_acc(dataframe, f_vals):
       for f in f_vals:
              print("Accuracy with f" + str(f) + " missing: ", end='')
              print(calc_lr_without_feature(dataframe, f))

calc_f_missing_acc(bn_df, [1,2,3,4])

# Question 5.2-5 : See supplemental documentation