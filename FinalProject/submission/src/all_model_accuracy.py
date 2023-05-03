
"""
Divya Thomas 
Class: CS 677
Date: 4/12/2023
Description of Problem (just a 1-2 line summary!): 
This is a file that calculates the accuracy counts for each classifier with the same split data
"""
import initial_analysis as ia
import knn
import l_kernel_svm as lk
import logistic_regression as lr
import naive_bayesian as nb
import consolidated as c

# Calculate the accuracy counts for each classifier with the same split data

df= ia.dataset()
split_data = ia.split(df)
# knn where k = 11 ( best k ) 
knn_pred = knn.knn_classifier(split_data[0], split_data[1], [11]) 
knn.print_counts(knn_pred[1][0])
print()

lk_pred = lk.linear_kernel(split_data[0], split_data[1])
lk.print_counts(lk_pred)
print()

lr_pred = lr.lr_classifier(split_data[0], split_data[1])
lr.print_counts(lr_pred)
print()

nb_pred = nb.predict_nb(split_data[0], split_data[1])
nb.print_counts(nb_pred)
print()

c_pred = c.consolidated_classifier(split_data[0], split_data[1])
c.print_counts(c_pred)
print()