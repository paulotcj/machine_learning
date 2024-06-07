print('----------------------------------------------')
print('Grid Search')


print('----------------------------------------------')
print('Importing the libraries')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


print('----------------------------------------------')
print('Importing the dataset')
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


print('----------------------------------------------')
print('Splitting the dataset into the Training set and Test set')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


print('----------------------------------------------')
print('Feature Scaling')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


print('----------------------------------------------')
print('Training the Kernel SVM model on the Training set')
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)


print('----------------------------------------------')
print('Making the Confusion Matrix')
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


print('----------------------------------------------')
print('Applying k-Fold Cross Validation')

# Note: K-Fold Cross Validation is a technique used to evaluate the performance of a machine learning model.
#     It is not doing any training, it is just evaluating the model.

from sklearn.model_selection import cross_val_score


accuracies = cross_val_score(
    estimator = classifier, # the classifier model we want to evaluate
    X = x_train, 
    y = y_train, 
    cv = 10, # 10 is the default value, and it means that we are going to use 10 folds
    n_jobs=-1 #n_jobs=-1 means that we are going to use all the processors of the computer
)


print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

print(f'\nAccuracies list: {accuracies}')
print('----')
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
print('\n')
print('----------------------------------------------')
print('Note: we have 4 types of combinations of accuracies and standard deviations\n')
print('    - Low accuracy and low standard deviation: the model is too simple. This means the result missed the target but they are all clustered together. This overall is bad you are consistently hitting one group area but missing the bullseye\n')
print('    - Low accuracy and high standard deviation: the model is too simple and too sensitive, capturing noise in the data. This means the result will miss the target and be scattered. This is the worst case scenario. Your predictions are all over the place and you are consistently missing the target.\n')
print('    - High accuracy and high standard deviation: the model is too sensitive and it is capturing noise. Not ideal but not the worst. This means the result is hitting the target but is scattered.\n')
print('    - High accuracy and low standard deviation: the model is just right. This means the result is hitting the target and is clustered together.\n\n')



print('----------------------------------------------')
print('Applying Grid Search to find the best model and the best parameters')
from sklearn.model_selection import GridSearchCV
parameters = [
    {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
    {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
]



grid_search = GridSearchCV(
    estimator = classifier,
    param_grid = parameters,
    scoring = 'accuracy',
    cv = 10,
    n_jobs = -1
)


grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)


print('----------------------------------------------')
print('Visualising the Training set results')
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


print('----------------------------------------------')
print('Visualising the Test set results')
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()