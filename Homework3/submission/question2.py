"""
Divya Thomas 
Class: CS 677
Date: 4/1/2023
Homework Problem #2
Description of Problem (just a 1-2 line summary!):
Use a training set of the data to predict the other half using a simple classifier,
then calculate the accuracy of it. 
"""
import os
import question1 as q1
import seaborn as sns
from sklearn.model_selection import train_test_split

bn_df = q1.create_pandas_dataframe()


print ("\n--Question 2.1--")
# Question 2.1 - Create a pairplot with the training data, one for each class
#split dataset 50/50 into a training and testing set 
train,test = train_test_split(bn_df, test_size=0.5)

def create_plot(df, filename):
    input_dir = os.path.abspath(__file__) + '\..\datasets'
    bn_file = os.path.join(input_dir, filename)
    sns.pairplot(df, hue='color', x_vars=["f1", "f2", "f3", "f4"], y_vars=["f1", "f2", "f3", "f4"]).savefig(bn_file)
#pairplot for all training data 
create_plot(train, "all_bills.pdf")
create_plot(train[train['class'] == 0], "good_bills.pdf")
create_plot(train[train['class'] == 1], "fake_bills.pdf")

print("See plots created under datasets directory.")

# see the generated plots in the datasets directory

# Question 2.2 - Comparisions are explained in the summary word document

print ("\n--Question 2.3--")
# Question 2.3 - Use the classifiers determined above to predict test data outcomes
def predict_validity(df):
    predictions = [] # to hold all the predictions
    #iterate through the dataframe
    for index, row in df.iterrows():
        # check using the classifiers
        if (row['f1'] >= 0) and (row['f2'] >= 5) and (row['f3'] <= 6):
            predictions.append("green") # good bill prediction
        else:
            predictions.append("red") # fake bill prediction

    # add the prediction col to the df
    df['prediction'] = predictions 

    return df 

predict_validity(test)
print(test)

print ("\n--Question 2.4--")
# Question 2.4 - Calculate the true/false negatives/positives of the predictions
def calc_label_accuracies(df):
    # get the list of true labels 
    true_list = df['color'].tolist()
    # get the list of the predicted labels
    label_list = df['prediction'].tolist()

    #set up counters for each scenario
    tp_count = 0
    fp_count = 0
    tn_count = 0
    fn_count = 0

    for i in range(len(label_list)):
        if true_list[i] == 'green':
            if label_list[i] == 'green': #true positive
                tp_count += 1
            elif label_list[i] == 'red': #false negative
                fn_count += 1
        elif true_list[i]   == 'red': 
            if label_list[i] == 'red': #true negative
                tn_count += 1
            elif label_list[i] == 'green': #false positive
                fp_count += 1
    count_list = [tp_count, fn_count, tn_count, fp_count]            
    return count_list

def calc_tpr(count_list):
    # TPR = TP/(TP+FN)
    tp = count_list[0]
    fn = count_list[1]
    #based off of calc_label_accuracies function

    tpr = tp / (tp + fn)
    return tpr

def calc_tnr(count_list):
    # TNR = TN/(TN+FP)
    tn = count_list[2]
    fp = count_list[3]
    #based off of calc_label_accuracies function

    tnr = tn / (tn + fp)
    return tnr

def accuracy(dataframe):
    # get the list of true labels 
    true_list = dataframe['color'].tolist()
    # get the list of the predicted labels 
    label_list = dataframe["prediction"].tolist()
    # get the total count of the predicted labels 
    tot_count = len(label_list)
    # set a counter for the correct predictions
    success_count = 0

    #match up each index in true_list with label_list and increment success count if they match
    for i in range(0, tot_count):
        if (true_list[i] == label_list[i]):
            success_count += 1
    # return the percentage of success
    return (success_count / tot_count) * 100

count_list = calc_label_accuracies(test)
acc = accuracy(test)
tpr = calc_tpr(count_list)
tnr = calc_tnr(count_list)
print('-Counts-')
print('TP: %d \nFP: %d \nTN: %d \nFN: %d \nAccuracy: %.2f \nTPR: %.2f \nTNR: %.2f' 
      % (count_list[0], count_list[3], count_list[2], count_list[1], acc, tpr, tnr))

# Question 2.5-6 : See supplemental documentation
