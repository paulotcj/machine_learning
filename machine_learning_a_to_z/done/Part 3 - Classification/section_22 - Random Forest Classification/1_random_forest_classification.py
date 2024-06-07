print('----------------------------------------------')
print('Random Forest Classification')

print('----------------------------------------------')
print('Import the libraries')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('----------------------------------------------')
print('Import the dataset')
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print('x (sample 5 first rows):')
print(x[0:5])
print('----')
print('y (sample 5 first rows):')
print(y[0:5])

print('----------------------------------------------')
print('Split the dataset into the Training set and Test set')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
print('x_train:')
print(x_train)
print('----')
print('y_train:')
print(y_train)
print('----')
print('x_test:')
print(x_test)
print('----')
print('y_test:')
print(y_test)

print('----------------------------------------------')
print('Feature Scaling')

print('    Please note we have 2 columns - Age and Estimated Salary')
print('----')
print('x_train (sample 5 first rows):')
print(x_train[0:5])
print('----')
print('x_test (sample 5 first rows):')
print(x_test[0:5])
print('----')
print('Scaling the x_train and x_test data')
print('----')


from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.transform(x_test)

print('x_train (sample 5 first rows):')
print(x_train[0:5])
print('----')
print('x_test (sample 5 first rows):')
print(x_test[0:5])

print('----------------------------------------------')
print('Train the Random Forest Classification model on the Training set')

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 40, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)


print('----------------------------------------------')
print('Predict a new result')
predict_age = 30
predict_salary = 87_000
print(f'  What to Predict: Age {predict_age}, Salary {predict_salary}')
predict_purchase = classifier.predict( standard_scaler.transform([[predict_age, predict_salary]]) )
print(f'  Purchase Prediction: {predict_purchase[0]}')

print('----------------------------------------------')
print('Predict the Test set results')
y_pred = classifier.predict(x_test)
print(
    np.concatenate(# reshaped to have as many rows as necessary and one column 
        (
            y_pred.reshape(len(y_pred),1), # as many rows as necessary and one column
            y_test.reshape(len(y_test),1)

        ),
        1 # axis=1 (columns) on which they will be joined
    )
)

print('----------------------------------------------')
print('Create the Confusion Matrix')
#to understand the code below:
# cm_result = array([[TN, FP],
#                    [FN, TP]])
#
# Where:
#
# TN (true negatives) - the model correctly predicted the negative class,
# FP (false positives) - the model incorrectly predicted the positive class,
#
# FN (false negatives) - the model incorrectly predicted the negative class
# TP (true positives) - the model correctly predicted the positive class

from sklearn.metrics import confusion_matrix, accuracy_score
cm_result = confusion_matrix(y_test, y_pred)
accuracy_score_result = accuracy_score(y_test, y_pred)

print('Confusion Matrix results:')
print(cm_result)

print('----')
print(f'    True Negatives: {cm_result[0][0]} - False Negatives: {cm_result[1][0]}')
print(f'    True Positives: {cm_result[1][1]} - False Positives: {cm_result[0][1]}')
print('----')
print('Accuracy Score:')
print(accuracy_score_result)

print('----------------------------------------------')
print('Visualize the Training set results')
from matplotlib.colors import ListedColormap
x_set, y_set = standard_scaler.inverse_transform(x_train), y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 10, stop = x_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = x_set[:, 1].min() - 1000, stop = x_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(x1, x2, classifier.predict(standard_scaler.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

print('----------------------------------------------')
print('Visualising the Test set results')
from matplotlib.colors import ListedColormap
x_set, y_set = standard_scaler.inverse_transform(x_test), y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 10, stop = x_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = x_set[:, 1].min() - 1000, stop = x_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(x1, x2, classifier.predict(standard_scaler.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()