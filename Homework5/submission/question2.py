"""
Divya Thomas 
Class: CS 677
Date: 4/15/2023
Homework Problem #2
Description of Problem (just a 1-2 line summary!):
Perform predictions of the NSP_CLASS using the Naive Bayesian classifier
and analyze its performance.
"""
import question1 as q1
import question5 as q5
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


df = q1.create_pandas_dataframe()

# Question 2.1 - Split df up into train and test data (X:['MSTV', 'Width', 'Mode', 'Variance', 'NSP'], Y : NSP_CLASS)
# then use Naive Bayesian NB classifier to predict class values for test data 
print ("\n--Question 2.1--")
# split test data 50/50 consistently
def split(df):
    x = df[['MSTV', 'Width', 'Mode', 'Variance']]
    y = df['NSP_CLASS']
    x_train,x_test,y_train,y_test = train_test_split(
        x, 
        y, 
        test_size=0.5, 
        random_state=42)
    
    train = (x_train, y_train)
    test = (x_test, y_test)
    return (train, test)

# Use Naive Bayesian NB classifier to predict class values for test data
def predict_nb(train, test):

    # Fit a Naive Bayesian classifier to the training data
    nb = GaussianNB()
    nb.fit(train[0], train[1])

    #predict test data classes with the model
    pred = nb.predict(test[0])

    #create a copy of the test data with the added predictions
    test_copy = test[0].copy()
    test_copy['Actual'] = test[1]
    test_copy['Prediction'] = pred

    return test_copy

split_data = split(df)
results = predict_nb(split_data[0], split_data[1])
print("NB Classifier Results: \n", results)

# Question 2.2 - Calculate the accuracy of the classifier
print ("\n--Question 2.2--")
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

acc = accuracy(results)
print ("Accuracy: %.2f%%" % acc)

# Question 2.3 - Construct the confusion matrix for this
print ("\n--Question 2.3--")
def compute_confusion_matrix(df):
    # compute the confusion matrix
    cm = confusion_matrix(df['Actual'], df['Prediction'])
    return cm

cm = compute_confusion_matrix(results)
print("Confusion Matrix: \n", cm)

def get_split():
    return split_data


# For Question 5
counts = q5.calc_label_accuracies(results)
tpr = q5.calc_tpr(counts)
tnr = q5.calc_tnr(counts)

def print_counts():
    print("\n--NB Counts (Q5)--")
    print("TP: %d \nFN: %d \nTN: %d \nFP: %d \nTPR: %.2f \nTNR: %.2f \nAccuracy: %.2f%%" % (counts[0], counts[1], counts[2], counts[3], tpr, tnr, acc))

print_counts()