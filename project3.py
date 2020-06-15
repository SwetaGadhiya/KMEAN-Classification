import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from munkres import Munkres, print_matrix
import sys

# This function will format lables to compare them with actual class labels
# When reading the file data, labels got converted into 1, 1.1, 1.2, .... so on
# So this function will convert them again in actual labels i.e, 1, 1, 1, .... 2, 2, 2.....
def formatLables(lables):
    formattedLables = []
    for data in lables:
        if (data.find('.') != -1):
            formattedLables.append(int(data[0:data.find('.')]))
        else:
            formattedLables.append(int(data))
    return formattedLables

# This function takes confusion matrix and number of instance per class as input and calculate accuracy
def findAccuracy(matrix, image_per_class):
    sum = 0
    for i in range(len(matrix)):
        sum = sum + matrix[i][i]
    return sum / (image_per_class * len(matrix))

# This function will reorder the confusion matrix by using hangerian method for bipertite matching.
def reorderMatrix(matrix):
    (total, indexes) = hangerian(matrix)
    reorderedMatrix = []
    rows, cols = (len(matrix), len(matrix))
    for i in range(rows):
        reorderedMatrix.append([])
        for j in range(cols):
            reorderedMatrix[i].append(0)
    reorderedMatrix = np.array(reorderedMatrix)
    for ind in indexes:
        reorderedMatrix[:,[ind[0]]] = matrix[:,[ind[1]]]
    return reorderedMatrix



def hangerian(matrix):
    cost_matrix = []
    for row in matrix:
        cost_row = []
        for col in row:
            cost_row += [sys.maxsize - col]
        cost_matrix += [cost_row]

    m = Munkres()
    indexes = m.compute(cost_matrix)
    # print_matrix(matrix, msg='Highest profit through this matrix:')
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        # print(f'({row}, {column}) -> {value}')

    # print(f'total profit={total}')
    return (total, indexes)

# This function is performing Project 3 - Task 1
def task_one():
    # Read data from input file
    df = pd.read_csv('ATNTFaceImages.txt').T
    # Initiate Kmean
    kmeans = KMeans(n_clusters=40, init='k-means++', max_iter=3000, n_init=20, random_state=0)
    # Process the data by running KMean on them
    kmeans.fit(np.array(df.values))
    print("Clustering result (centroids): ",kmeans.cluster_centers_)
    print("K-MEAN Loss: ",kmeans.inertia_)
    # List of actual class labels
    actual = formatLables(np.array(df.index))
    classified_class_on_train_instances = []
    for pred_val in kmeans.labels_:
        classified_class_on_train_instances.append(pred_val + 1)
    # Calculate confusion matrix
    results = confusion_matrix(actual, classified_class_on_train_instances)
    print("----------------- Confusion Matrix -----------------")
    print(results)
    print('-----------------Actual Accuracy -----------------')
    print(findAccuracy(results, 10)*100)
    reordered_matrix = reorderMatrix(results)
    print('----------------- Reordered Matrix -----------------')
    print(reordered_matrix)
    print('----------------- Reordered Matrix Accuracy -----------------')
    print(findAccuracy(reordered_matrix, 10)*100)

# This function is performing Project 3 - Task 2
def task_two():
    # Read data from input file
    df = pd.read_csv('HandWrittenLetters.txt').T
    # Initiate Kmean
    kmeans = KMeans(n_clusters=26, init='k-means++', max_iter=3000, n_init=20, random_state=0)
    # Process the data by running KMean on them
    kmeans.fit(np.array(df.values))
    print("Clustering result (centroids): ",kmeans.cluster_centers_)
    print("K-MEAN Loss: ",kmeans.inertia_)
    # List of actual class labels
    actual = formatLables(np.array(df.index))
    classified_class_on_train_instances = []
    for pred_val in kmeans.labels_:
        classified_class_on_train_instances.append(pred_val + 1)
    # Calculate confusion matrix
    results = confusion_matrix(actual, classified_class_on_train_instances)
    print("----------------- Confusion Matrix -----------------")
    print(results)
    print('-----------------Actual Accuracy -----------------')
    print(findAccuracy(results, 39)*100)
    reordered_matrix = reorderMatrix(results)
    print('----------------- Reordered Matrix -----------------')
    print(reordered_matrix)
    print('----------------- Reordered Matrix Accuracy -----------------')
    print(findAccuracy(reordered_matrix, 39)*100)


def Main():
    print('*********************************************    Project3 A.    *********************************************')
    task_one()
    print('*********************************************    Project3 B.    *********************************************')
    task_two()

if __name__ == '__main__':
    Main()
