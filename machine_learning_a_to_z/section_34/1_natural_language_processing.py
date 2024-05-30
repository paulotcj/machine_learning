print('----------------------------------------------')
print('Natural Language Processing')

print('----------------------------------------------')
print('Importing the libraries')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('----------------------------------------------')
print('Importing the dataset')


dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
# The quoting parameter controls when quotes should be recognized. 
#  There are four options you can choose from (0, 1, 2, 3). The option 3 tells 
#  pandas to ignore quotes altogether. This can be useful when reading a file 
#  that uses quotes inconsistently

# csv.QUOTE_ALL (value is 1): Quote everything, regardless of type.
# csv.QUOTE_MINIMAL (value is 0): Quote fields with special characters (anything that would confuse a CSV parser). This is the default behavior.
# csv.QUOTE_NONNUMERIC (value is 2): Quote all fields that are not integers or floats. When used with the writer, non-numeric data will be quoted. When used with the reader, non-quoted data will be converted to floats.
# csv.QUOTE_NONE (value is 3): Do not quote anything on output. When used with the reader, quote characters are treated as regular characters.

print(f'Dataset first 5 rows')
print(dataset[0:5])

print('----------------------------------------------')
print('Cleaning the texts')

import re #regular expression
import nltk #Natural Language Toolkit (NLTK)

#--------------
# little work around for when the SSL fails
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
#--------------

# Stop words are a set of commonly used words in a language. Examples of stop words in English 
#  are "a", "and", "the", "in", etc. They are often removed from texts as they do not carry much 
#  meaning and are usually not needed in Natural Language Processing tasks.
# The nltk.download() function is a built-in f
# unction in NLTK used to download additional resources, 
#  such as corpora, grammars, models, etc.

nltk.download('stopwords') #this will be done only once

#--------------
# The PorterStemmer is an implementation of the Porter stemming algorithm, which is a widely used 
#   method of stemming English words. It's a process of heuristic cleaning to remove common endings 
#   from words to help in the task of information retrieval. The algorithm has five phases of word 
#   reduction, each with its own set of mapping rules. For example, it might transform "running" 
#   into "run", "happiness" into "happi", and "argument" into "argu".

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
#--------------
for i in range(0, dataset.shape[0]): # dataset.shape[0] -> num of rows - in this case 10_000

  #replace anything that is not alphanumeric with spaces
  #  use the Review column, at row 'i'
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

# end of for i in range(0, dataset.shape[0]):
#--------------

print(corpus)

print('----------------------------------------------')
print('Creating the Bag of Words model')
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

print('----------------------------------------------')
print('Splitting the dataset into the Training set and Test set')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

print('----------------------------------------------')
print('Training the Naive Bayes model on the Training set')
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

print('----------------------------------------------')
print('Predicting the Test set results')
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

print('----------------------------------------------')
print('Making the Confusion Matrix')
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)