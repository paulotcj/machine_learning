print('----------------------------------------------')
print('Principal Component Analysis (PCA)')

print('----------------------------------------------')
print('Importing the libraries')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('----------------------------------------------')
print('Importing the dataset')
dataset = pd.read_csv('Wine.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print('----------------------------------------------')
print('Splitting the dataset into the Training set and Test set')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print('----------------------------------------------')
print('Feature Scaling')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print('----------------------------------------------')
print('Applying PCA')

print('x_train first rows')
print(x_train[:5])
print('----')
print('x_test first rows')
print(x_test[:5])

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

print('After PCA')
print('x_train first rows')
print(x_train[:5])
print('----')
print('x_test first rows')
print(x_test[:5])


print('----------------------------------------------')
print('Training the Logistic Regression model on the Training set')
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

print('----------------------------------------------')
print('Making the Confusion Matrix')
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(x_test)
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# accuracy_score(y_test, y_pred)

cm_result = confusion_matrix(y_test, y_pred)
print(cm_result)

accuracy_score_result = accuracy_score(y_test, y_pred)

print('----')
print(f'    True Negatives: {cm_result[0][0]} - False Negatives: {cm_result[1][0]}')
print(f'    True Positives: {cm_result[1][1]} - False Positives: {cm_result[0][1]}')
print('----')
print('Accuracy Score:')
print(accuracy_score_result)



print('----------------------------------------------')
print('Visualising the Training set results')
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

print('----------------------------------------------')
print('Visualising the Test set results')
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()