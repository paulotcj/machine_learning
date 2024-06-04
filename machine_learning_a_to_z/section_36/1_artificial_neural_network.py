print('----------------------------------------------')
print('Artificial Neural Network')

print('----------------------------------------------')
print('Importing the libraries')

import numpy as np
import pandas as pd
# import tensorflow as tf

# print(tf.__version__)


print('----------------------------------------------')
print('Part 1 - Data Preprocessing')

print('----------------------------------------------')
print('Importing the dataset')

dataset = pd.read_csv('Churn_Modelling.csv')
# the columns are: RowNumber,CustomerId,Surname,CreditScore,Geography,Gender,Age,Tenure,
#   Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited
x = dataset.iloc[:,3:-1].values #we want from col 3 (CreditScore) to the second last column (EstimatedSalary)
y = dataset.iloc[:,-2].values #we want the last column (Exited)

print(f'x shape: {x.shape}')
print(f'x first 5 rows')
print(x[0:5])
print('----')
print(f'y shape: {y.shape}')
print(f'y first 5 rows')
print(y[0:5])

exit()

print('----------------------------------------------')
print('Encoding categorical data')

print('----------------------------------------------')
print('Label Encoding the "Gender" column')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])
print(x)

print('----------------------------------------------')
print('One Hot Encoding the "Geography" column')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)


print('----------------------------------------------')
print('Splitting the dataset into the Training set and Test set')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print('----------------------------------------------')
print('Feature Scaling')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print('----------------------------------------------')
print('Part 2 - Building the ANN')

print('----------------------------------------------')
print('Initializing the ANN')
ann = tf.keras.models.Sequential()

print('----------------------------------------------')
print('Adding the input layer and the first hidden layer')
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

print('----------------------------------------------')
print('Adding the second hidden layer')
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

print('----------------------------------------------')
print('Adding the output layer')
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

print('----------------------------------------------')
print('Part 3 - Training the ANN')

print('----------------------------------------------')
print('Compiling the ANN')
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

print('----------------------------------------------')
print('Training the ANN on the Training set')
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

print('----------------------------------------------')
print('Part 4 - Making the predictions and evaluating the model')

print('----------------------------------------------')
print('Predicting the result of a single observation')

"""
Homework:
Use our ANN model to predict if the customer with the following informations will leave the bank: 
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000
So, should we say goodbye to that customer?

Solution:
"""

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

"""
Therefore, our ANN model predicts that this customer stays in the bank!
Important note 1: Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.
Important note 2: Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.
"""

print('----------------------------------------------')
print('Predicting the Test set results')
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

print('----------------------------------------------')
print('Making the Confusion Matrix')
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)