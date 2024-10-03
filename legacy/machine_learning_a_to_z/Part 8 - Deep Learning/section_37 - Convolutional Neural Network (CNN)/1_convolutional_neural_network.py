print('----------------------------------------------')
print('Convolutional Neural Network')

print('----------------------------------------------')
print('Importing the libraries')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

print(tf.__version__)

print('----------------------------------------------')
print('Part 1 - Data Preprocessing')

print('----------------------------------------------')
print('Preprocessing the Training set')
print('  Note the output informing how many classes were found is based on the number of folders in the training_set')

# Image Augmentation - we modify the original images so the CNN doesn't overlearn on
#  the existing images

train_datagen = ImageDataGenerator(
    rescale = 1./255, #this parameter refers to feature scalling - it will divide the value of each pixel by 255
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)


#this will be used directly
training_set = train_datagen.flow_from_directory(
    'dataset/training_set', #path
    target_size = (64, 64), # this will be 64x64 pixels - the size of the image sent to the CNN
    batch_size = 32,
    class_mode = 'binary'#can be binary or categorical , but since we are deciding between cat and dog, it will be binary
)


print('----------------------------------------------')
print('Preprocessing the Test set')

#we don't need to apply the other transformations to the test set, as we do not have to care for image augmentation
test_datagen = ImageDataGenerator(rescale = 1./255)  #feature scalling - divide the value of each pixel by 255


#this will be used directly
test_set = test_datagen.flow_from_directory(
    'dataset/test_set', #path
    target_size = (64, 64), #image size 
    batch_size = 32, #number of images in each batch
    class_mode = 'binary' #binary or categorical, in this case binary
) 

print('----------------------------------------------')
print('Part 2 - Building the CNN')

print('----------------------------------------------')
print('Initialising the CNN')

cnn = tf.keras.models.Sequential()

print('----------------------------------------------')
print('Step 1 - Convolution')

# note about the filters - 32 is a common choice for the number of filters in the first layer of a CNN
cnn.add(
    tf.keras.layers.Conv2D(
        filters = 32, # specifies the number of output filters in the convolution, or in other words, the number of feature detectors
        kernel_size = 3, # height and width of the 2D convolution window - kernel size 3 means the network will use 3x3 filters
        activation = 'relu', 
        input_shape = [64, 64, 3] # 64 pixels by 64 pixels, with 3 color channels
    ) 
)


print('----------------------------------------------')
print('Step 2 - Pooling')

#max pooling: we gather the max value of a group of pixels, in this example a 2x2 group of pixels, and we set the result as the max
#  value of the group.
cnn.add(
    tf.keras.layers.MaxPool2D(
        pool_size=2, #size of the pooling window, in this case 2x2 pixels
        strides=2 #moving the pooling window by 2 pixels (this is the step size of the pooling window)
    )
)

print('----------------------------------------------')
print('Adding a second convolutional layer')

#note about the filters - we are using the same architecture as before, therefore sticking to 32 filters
cnn.add(
    tf.keras.layers.Conv2D(
        filters=32, 
        kernel_size=3, 
        activation='relu'
    )
)

cnn.add( #add another pooling layer
    tf.keras.layers.MaxPool2D(
        pool_size=2, 
        strides=2
        )
)

print('----------------------------------------------')
print('Step 3 - Flattening')

cnn.add(
    tf.keras.layers.Flatten() #this will transform the 2D array into a 1D array
) 

print('----------------------------------------------')
print('Step 4 - Full Connection')

# for units we selected 128, but this is a hyperparameter that can be tuned. Experimentally, 128 is a good starting point
cnn.add(
    tf.keras.layers.Dense(
        units = 128, 
        activation = 'relu'
    )
)


print('----------------------------------------------')
print('Step 5 - Output Layer')

#only 1 output
cnn.add(
    tf.keras.layers.Dense(
        units=1, 
        activation='sigmoid'
    )
)

print('----------------------------------------------')
print('Part 3 - Training the CNN')

print('----------------------------------------------')
print('Compiling the CNN')

# for loss, we have only two categories, so we use binary_crossentropy. Also note that the output layer has only 1 unit, hence binary
cnn.compile(
    optimizer = 'adam', # Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
    loss = 'binary_crossentropy', 
    metrics = ['accuracy']
)



print('----------------------------------------------')
print('Training the CNN on the Training set and evaluating it on the Test set')
cnn.fit(
    x = training_set, 
    validation_data = test_set, 
    epochs = 25
)





print('----------------------------------------------')
print('Part 4 - Making a single prediction')


import numpy as np
from keras.preprocessing import image #type: ignore

# this is a pure image. note that we don't change the image in any way, as we are not training the model
test_image = image.load_img(
    'dataset/single_prediction/cat_or_dog_1.jpg', 
    target_size = (64, 64) #size we will convert the image to
)

test_image = image.img_to_array(test_image) # this is a numpy array with dimensions of (64, 64, 3)
#   64x64 pixels and 3 channels (RGB)





test_image = np.expand_dims(test_image, axis = 0) # this is a numpy array with dimensions of (1, 64, 64, 3)
#  1 array batch, containing a single image (64 x 64 x 3)

result = cnn.predict(test_image)

# print(training_set.class_indices)
# if result[0][0] == 1:
#   prediction = 'dog'
# else:
#   prediction = 'cat'
#--------------------------

print(f'\n\n training_set.class_indices : {training_set.class_indices}\n\n')


for key, value in training_set.class_indices.items():
  if value == result:
    print(f'The image is predicted to belong to the category: {key}')
    break

print('\n\n')