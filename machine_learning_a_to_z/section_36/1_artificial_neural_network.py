print('----------------------------------------------')
print('Artificial Neural Network')

print('----------------------------------------------')
print('Importing the libraries')

import numpy as np
import pandas as pd





print('----------------------------------------------')
print('Part 1 - Data Preprocessing')

print('----------------------------------------------')
print('Importing the dataset')

dataset = pd.read_csv('Churn_Modelling.csv')
# the columns are: RowNumber,CustomerId,Surname,CreditScore,Geography,Gender,Age,Tenure,
#   Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited
x = dataset.iloc[:,3:-1].values #we want from col 3 (CreditScore) to the second last column (EstimatedSalary)
y = dataset.iloc[:,-2].values #we want the last column (Exited)

print(f'x rows: {x.shape[0]}, x cols: {x.shape[1]}')
print(f'x first 10 rows')
print(x[0:10])
print('----')
print(f'y shape: {y.shape}')
print(f'y first 5 rows')
print(y[0:5])


print('----------------------------------------------')
print('Encoding categorical data')

print('----------------------------------------------')
print('Label Encoding the "Gender" column')


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
x[:, 2] = label_encoder.fit_transform( x[:, 2] )
print(f'x first 10 rows')
print(x[0:10])

print('----------------------------------------------')
print('One Hot Encoding the "Geography" column')

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# The [1] means we will apply the OneHotEncoder to the second column of the matrix of features x.
# remainder='passthrough' parameter specifies what should be done with the columns that are 
#   not explicitly selected for transformation. In this case, 'passthrough' means that the 
#   remaining columns should be left as they are and concatenated with the output of the transformer.
column_transformer = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(column_transformer.fit_transform(x))

# for row in x[0:10]:
#     print('-----')
#     list_row = []
#     for col in row:
#         list_row.append(col)
        
#     print(list_row)


print(x[0:10])


print('----------------------------------------------')
print('Splitting the dataset into the Training set and Test set')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2 , random_state = 0)
print(f'x_train')
print(x_train[0:10])
print('-----')
print(f'x_test')
print(x_test[0:10])
print('-----')
print(f'y_train')
print(y_train[0:10])
print('-----')
print(f'y_test')
print(y_test[0:10])
print('-----')

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print('----------------------------------------------')
print('Feature Scaling')

from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.transform(x_test)

print(f'x_train rows length : {x_train.shape[1]}')
print(f'x_train')
print(x_train[0:10])
print('-----')
print(f'x_test')
print(x_test[0:10])
print('-----')

# for row in x[0:10]:
#     print(f'-----  row len: {len(row)}')
#     list_row = []
#     for col in row:
#         list_row.append(col)
        
#     print(list_row)




print('----------------------------------------------')
print('Part 2 - Building the ANN')

print('----------------------------------------------')
print('Initializing the ANN')

import tensorflow as tf
print(tf.__version__)

# The Sequential model is a linear stack of layers. You can create a 
#   Sequential model by passing a list of layer instances to the constructor, 
#   or by simply adding layers via the .add() method.
ann = tf.keras.models.Sequential()


print('----------------------------------------------')
print('Adding hidden layers')


# When you create and train a model using Keras, the input shape is inferred from the 
#   data automatically.
#
# When you call fit on a model, Keras looks at the number of features in your input 
#   data (in this case, x_train) and automatically adjusts the input shape of your model. 
# So, if your x_train has 12 columns, the input layer of your model will automatically 
#   expect inputs with 12 features.


# 6 neurons in this layer
# The activation function we will use is the rectifier function 'relu'
ann.add( tf.keras.layers.Dense(units = 6, activation = 'relu') )

# 6 neurons in this layer
# The activation function we will use is the rectifier function 'relu'
ann.add( tf.keras.layers.Dense(units = 6 , activation = 'relu') ) 


# 6 neurons in this layer
# The activation function we will use is the rectifier function 'relu'
ann.add( tf.keras.layers.Dense(units = 6, activation = 'relu') )


#note: You can try a different number of neurons in the hidden layers, experimentally the
#  results don't change much.

print('----------------------------------------------')
print('Adding the output layer')

ann.add( tf.keras.layers.Dense(units = 1, activation = 'sigmoid') )

exit()

print('----------------------------------------------')
print('Part 3 - Training the ANN')

print('----------------------------------------------')
print('Compiling the ANN')
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

print('----------------------------------------------')
print('Training the ANN on the Training set')
ann.fit(x_train, y_train, batch_size = 32, epochs = 100)

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

print(ann.predict(standard_scaler.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

"""
Therefore, our ANN model predicts that this customer stays in the bank!
Important note 1: Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.
Important note 2: Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.
"""

print('----------------------------------------------')
print('Predicting the Test set results')
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

print('----------------------------------------------')
print('Making the Confusion Matrix')
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)