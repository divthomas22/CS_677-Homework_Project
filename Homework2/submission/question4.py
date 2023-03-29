"""
Divya Thomas 
Class: CS 677
Date: 3/24/2023
Homework Problem #4
Description of Problem (just a 1-2 line summary!):
Compute accuracy of the true/false positives and negatives for each stock
"""
import question1 as q1
import question2 as q2
import question3 as q3

tgt_df = q1.create_pandas_dataframe('TGT')
q1.add_true_label(tgt_df)
spy_df = q1.create_pandas_dataframe('SPY')
q1.add_true_label(spy_df)

tgt_w2_4 = q2.predict_w2_4(tgt_df)
spy_w2_4 = q2.predict_w2_4(spy_df)

tgt_we = q3.ensemble(tgt_w2_4)
spy_we = q3.ensemble(spy_w2_4)

# Question 4.1-4 - Function to calculate the false positive/negative and true positive/negative accuracies
def calc_label_accuracies(df, w):
    # get the list of true labels 
    true_list = df['True Label'].tolist()
    # get the list of the predicted labels for the w param passed
    label_list = df["W" + str(w)].tolist()

    #set up counters for each scenario
    tp_count = 0
    fp_count = 0
    tn_count = 0
    fn_count = 0

    for i in range(len(label_list)):
        if true_list[i] == '+':
            if label_list[i] == '+': #true positive
                tp_count += 1
            elif label_list[i] == '-': #false negative
                fn_count += 1
        elif true_list[i]   == '-': 
            if label_list[i] == '-': #true negative
                tn_count += 1
            elif label_list[i] == '+': #false positive
                fp_count += 1
    count_list = [tp_count, fn_count, tn_count, fp_count]            
    return count_list

tgt_w2_counts = calc_label_accuracies(tgt_we, 2)
tgt_w3_counts = calc_label_accuracies(tgt_we, 3)
tgt_w4_counts = calc_label_accuracies(tgt_we, 4)
tgt_we_counts = calc_label_accuracies(tgt_we, 'E')

spy_w2_counts = calc_label_accuracies(spy_we, 2)
spy_w3_counts = calc_label_accuracies(spy_we, 3)
spy_w4_counts = calc_label_accuracies(spy_we, 4)
spy_we_counts = calc_label_accuracies(spy_we, 'E')

print ("\n--Question 4.1--")
print ("TGT True Positives:")
print ("W2: ", tgt_w2_counts[0],
       "\nW3: ", tgt_w3_counts[0],
       "\nW4: ", tgt_w4_counts[0],
       "\nEnsemble: ", tgt_we_counts[0])
print ("\nSPY True Positives:")
print ("W2: ", spy_w2_counts[0],
       "\nW3: ", spy_w3_counts[0],
       "\nW4: ", spy_w4_counts[0],
       "\nEnsemble: ", spy_we_counts[0])

print ("\n--Question 4.2--")
print ("TGT False Positives:")
print ("W2: ", tgt_w2_counts[3],
       "\nW3: ", tgt_w3_counts[3],
       "\nW4: ", tgt_w4_counts[3],
       "\nEnsemble: ", tgt_we_counts[3])
print ("\nSPY False Positives:")
print ("W2: ", spy_w2_counts[3],
       "\nW3: ", spy_w3_counts[3],
       "\nW4: ", spy_w4_counts[3],
       "\nEnsemble: ", spy_we_counts[3])

print ("\n--Question 4.3--")
print ("TGT True Negatives:")
print ("W2: ", tgt_w2_counts[2],
       "\nW3: ", tgt_w3_counts[2],
       "\nW4: ", tgt_w4_counts[2],
       "\nEnsemble: ", tgt_we_counts[2])
print ("\nSPY True Negatives:")
print ("W2: ", spy_w2_counts[2],
       "\nW3: ", spy_w3_counts[2],
       "\nW4: ", spy_w4_counts[2],
       "\nEnsemble: ", spy_we_counts[2])

print ("\n--Question 4.4--")
print ("TGT False Negatives:")
print ("W2: ", tgt_w2_counts[1],
       "\nW3: ", tgt_w3_counts[1],
       "\nW4: ", tgt_w4_counts[1],
       "\nEnsemble: ", tgt_we_counts[1])
print ("\nSPY False Negatives:")
print ("W2: ", spy_w2_counts[1],
       "\nW3: ", spy_w3_counts[1],
       "\nW4: ", spy_w4_counts[1],
       "\nEnsemble: ", spy_we_counts[1])


print ("\n--Question 4.5--")
# Question 4.5 - Function to calculate the True Positive Rate based off of the count list provided
def calc_tpr(count_list):
    # TPR = TP/(TP+FN)
    tp = count_list[0]
    fn = count_list[1]
    #based off of calc_label_accuracies function

    tpr = tp / (tp + fn)
    return tpr

print ("TGT TPR:")
print ("W2: ", calc_tpr(tgt_w2_counts),
       "\nW3: ", calc_tpr(tgt_w3_counts),
       "\nW4: ", calc_tpr(tgt_w4_counts),
       "\nEnsemble: ", calc_tpr(tgt_we_counts))
print ("\nSPY TPR:")
print ("W2: ", calc_tpr(spy_w2_counts),
       "\nW3: ", calc_tpr(spy_w3_counts),
       "\nW4: ", calc_tpr(spy_w4_counts),
       "\nEnsemble: ", calc_tpr(spy_we_counts))
    
print ("\n--Question 4.6--")
# Question 4.6 - Function to calculate the True Negative Rate based off of the count list provided
def calc_tnr(count_list):
    # TNR = TN/(TN+FP)
    tn = count_list[2]
    fp = count_list[3]
    #based off of calc_label_accuracies function

    tnr = tn / (tn + fp)
    return tnr

print ("TGT TNR:")
print ("W2: ", calc_tnr(tgt_w2_counts),
       "\nW3: ", calc_tnr(tgt_w3_counts),
       "\nW4: ", calc_tnr(tgt_w4_counts),
       "\nEnsemble: ", calc_tnr(tgt_we_counts))
print ("\nSPY TNR:")
print ("W2: ", calc_tnr(spy_w2_counts),
       "\nW3: ", calc_tnr(spy_w3_counts),
       "\nW4: ", calc_tnr(spy_w4_counts),
       "\nEnsemble: ", calc_tnr(spy_we_counts))
    