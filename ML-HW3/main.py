#-------------------------------------------------------------------------
# AUTHOR: Suhuan Pan
# FILENAME: CS4210-SVM/ML-HW3/main.py
# SPECIFICATION: build multiple SVM classifiers based on the training data,
# compute its accuracy based on the testing data,
# then find out the best performance score and output its corresponding hyper parameter value
# FOR: CS 4210 - Assignment #3
# TIME SPENT: 3 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn import svm
import pandas as pd # import python interpreter package pandas
import csv
import numpy as np

# defining the hyper_parameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

####### 1. loading data #######
# reading the training data by using Pandas library
df = pd.read_csv('optdigits.tra', sep=',', header=None)

# getting the first 64 fields to create the
# feature training data and convert them to NumPy array
X_training = np.array(df.values)[:,:64]

# getting the last field to create the class
# training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1]

# reading the training data by using Pandas library
df = pd.read_csv('optdigits.tes', sep=',', header=None)

#getting the first 64 fields to create the
#feature testing data and convert them to NumPy array
# will be used in argument of predict()
X_test = np.array(df.values)[:,:64]

#getting the last field to create the class
#testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1]

# initialize the highest score and the index of its corresponding hyper_parameter
bestScore = 0.0000
h1 = 0
h2 = 0
h3 = 0
h4 = 0

# created 4 nested for loops that will iterate through the values of c, degree,
# kernel, and decision_function_shape
for i1 in range(len(c)):
    for i2 in range(len(degree)):
        for i3 in range(len(kernel)):
            for i4 in range(len(decision_function_shape)):
                
                ####### 2. Generating a Classifier #######
                # Create an SVM classifier that will test all combinations of c,
                # degree, kernel, and decision_function_shape.
                clf = svm.SVC(C=c[i1], degree=degree[i2], kernel=kernel[i3],
                              decision_function_shape=decision_function_shape[i4])

                ####### 3. Training the classifier #######
                ####### 4. Making the SVM prediction #######
                clf.fit(X_training, y_training) # Fit SVM to the training data
                y_predict = clf.predict(X_test) # X_test is 2-D array

                ####### 5. compute the accuracy #######
                matchCount = 0
                for i5 in range(len(y_predict)):
                    if y_predict[i5] == y_test[i5]:
                        matchCount += 1
                # end the for i5 loop

                accuracy = (matchCount / len(y_predict))
                
                # check if the calculated accuracy is higher than the previously one
                # calculated. If so, update the highest accuracy and print it together
                # with the SVM hyper_parameters.
                if accuracy > bestScore:
                    bestScore = accuracy
                    h1 = i1
                    h2 = i2
                    h3 = i3
                    h4 = i4

                bestScore = round(bestScore, 2) 
                print("Highest Score =", bestScore)
                print("Parameters: c =", c[h1], ", degree =", degree[h2], ", kernel =", kernel[h3],
                      ", decision_function_shape ='", decision_function_shape[h4], "'")

            # end the for i4 loop
    # end the for i1 loop

print("\n------ Final Result ------")
bestScore = round(bestScore, 2)
print("Highest Score =", bestScore)
print("Parameters: c =", c[h1], ", degree =", degree[h2], ", kernel =", kernel[h3],
      ", decision_function_shape ='", decision_function_shape[h4], "'")
