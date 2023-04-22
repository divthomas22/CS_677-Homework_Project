"""
Divya Thomas 
Class: CS 677
Date: 4/15/2023
Homework Problem #3
Description of Problem (just a 1-2 line summary!):
Perform predictions of the NSP_CLASS using the Decision Tree classifier
and analyze its performance.
"""
import question2 as q2
import question5 as q5
from sklearn.tree import DecisionTreeClassifier


# Question 3.1 - Split df up into train and test data (X:['MSTV', 'Width', 'Mode', 'Variance', 'NSP'], Y : NSP_CLASS)
# then use Decision Tree classifier to predict class values for test data 
print ("\n--Question 3.1--")
#get train and test data from q2
split_data = q2.get_split()

# Use the Decision Tree classifier to predict class values for test data
def predict_dt(train, test):

    # Fit a Decision Tree classifier to the training data
    dt = DecisionTreeClassifier()
    dt.fit(train[0], train[1])

    #predict test data classes with the model
    pred = dt.predict(test[0])

    #create a copy of the test data with the added predictions
    test_copy = test[0].copy()
    test_copy['Actual'] = test[1]
    test_copy['Prediction'] = pred

    return test_copy

results = predict_dt(split_data[0], split_data[1])
print("DT Classifier Results: \n", results)

# Question 3.2 - Calculate the accuracy of the classifier
print ("\n--Question 3.2--")
acc = q2.accuracy(results)
print ("Accuracy: %.2f%%" % acc)

# Question 3.3 - Construct the confusion matrix for this
print ("\n--Question 3.3--")
cm = q2.compute_confusion_matrix(results)
print("Confusion Matrix: \n", cm)

# Question 5 - print the TN, FN, TP, FP, TNR, and TPR
counts = q5.calc_label_accuracies(results)
tpr = q5.calc_tpr(counts)
tnr = q5.calc_tnr(counts)

def print_counts():
    print("\n--DT Counts (Q5)--")
    print("TP: %d \nFN: %d \nTN: %d \nFP: %d \nTPR: %.2f \nTNR: %.2f \nAccuracy: %.2f%%" % (counts[0], counts[1], counts[2], counts[3], tpr, tnr, acc))

print_counts()