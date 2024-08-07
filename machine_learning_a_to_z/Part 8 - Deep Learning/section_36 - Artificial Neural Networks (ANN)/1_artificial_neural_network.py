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
y = dataset.iloc[:,-1].values #we want the last column (Exited)

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




# # 6 neurons in this layer
# # The activation function we will use is the rectifier function 'relu'
# ann.add( tf.keras.layers.Dense(units = 6, activation = 'relu') )


#note: You can try a different number of neurons in the hidden layers, experimentally the
#  results don't change much.

print('----------------------------------------------')
print('Adding the output layer')

ann.add( tf.keras.layers.Dense(units = 1 , activation = 'sigmoid') )





print('----------------------------------------------')
print('Part 3 - Training the ANN')

print('----------------------------------------------')
print('Compiling the ANN')

# optimizer = 'adam' means that the Adam optimization algorithm is used. Adam is a 
#   popular choice because it combines the advantages of two other extensions of stochastic 
#   gradient descent: AdaGrad and RMSProp.
#
# loss = 'binary_crossentropy' -> In binary classification, we are trying to predict two 
#   possible outcomes, often denoted as 0 and 1. Cross-entropy is a measure of the difference 
#   between the model's predictions and the actual values. In the case of binary classification, 
#   we use binary cross-entropy as the loss function.
# The binary cross-entropy loss function is defined mathematically as the negative average of 
#   the log of the predicted probabilities for the actual classes. It is suitable for binary 
#   classification problems, and it's used when the output of the model is a probability that 
#   each input belongs to class 1.
# In the context of neural networks, the loss function is what the model tries to minimize 
#   during the training process. So, by specifying binary_crossentropy as the loss function, 
#   we're telling the model to minimize the binary cross-entropy between the predictions and 
#   the actual values.
#
# metrics parameter is optional and is used to specify the metric(s) that the model will 
#   use to evaluate its performance during training and testing. In this case, 
#   metrics = ['accuracy'] means that the model will use accuracy as its performance metric.

ann.compile( optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'] )



print('----------------------------------------------')
print('Training the ANN on the Training set')

# batch_size = 32 (for Stochastic Gradient Descent) means that the model will use 32 training 
#   examples at each step of the optimizer algorithm. This is a common choice for batch size, 
#   as it's a balance between computational efficiency and model performance.
#
# epochs = 100 means that the learning algorithm will make 100 passes through the training 
#   dataset.

ann.fit(x_train, y_train, batch_size = 32, epochs = 100)



print('----------------------------------------------')
print('Part 4 - Making the predictions and evaluating the model')

print('----------------------------------------------')
print('Predicting the result of a single observation')


# Use the ANN model to predict if the customer will leave the bank: 
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 years old
# Tenure: 3 years
# Balance: $ 60000
# Number of Products: 2
# Does this customer have a credit card? Yes
# Is this customer an Active Member: Yes
# Estimated Salary: $ 50000

var_predict_this = [
    [
        1, #France - yes
        0, #Spain - no
        0, #Germany - no
        600, #Credit Score
        1, # Male (0 = female, 1 = male)
        40, # Age
        3, # Tenure (years)
        60000, # Balance
        2, # Number of Products
        1, # Has Credit Card (0 = no, 1 = yes)
        1, # Is Active Member (0 = no, 1 = yes)
        50000 # Estimated Salary
    ]
]

var_prediction_result = ann.predict(
    standard_scaler.transform(var_predict_this)    
)

print(f'var_predict_this: {var_prediction_result[0][0] * 100}% chance of leaving the bank')
print(f'result = {var_prediction_result[0][0] > 0.5} - ( 0/False = stays , 1/True = leaves )')


print('----------------------------------------------')
print('Predicting the Test set results')

y_pred_chances = ann.predict(x_test)
y_pred = (y_pred_chances > 0.5)

print(f'y_pred_chances: % chance of leaving the bank')
print(y_pred_chances[0:10])
print('----')
print(f'y_pred result = ( 0/False = stays , 1/True = leaves )')
print(y_pred[0:10])
print('----')
print(f'[ prediction , actual ]')
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
# y_pred = ann.predict(x_test)
# y_pred = (y_pred > 0.5)


print('----------------------------------------------')
print('Making the Confusion Matrix')

from sklearn.metrics import confusion_matrix, accuracy_score
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