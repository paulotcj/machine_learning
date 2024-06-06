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
print('Natual Language Frameworks and Setup')

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
print('----------------------------------------------')
print('Stop Words')
ps = PorterStemmer() #this can be declared outside the loop
all_stopwords = stopwords.words('english')

all_stopwords.remove('not') #remove any other words deemed necessary - in this case not actually help with prediction - so we removed from the stop words
all_stopwords.remove('against')
all_stopwords.remove("through")
all_stopwords.remove("above")
all_stopwords.remove("below")
all_stopwords.remove('up')
all_stopwords.remove('down')
all_stopwords.remove('out')
all_stopwords.remove('off')
all_stopwords.remove('over')
all_stopwords.remove('under')
all_stopwords.remove('no')
all_stopwords.remove('nor')
all_stopwords.remove('don')
all_stopwords.remove("don't")
all_stopwords.remove("aren")
all_stopwords.remove("aren't")
all_stopwords.remove("couldn")
all_stopwords.remove("couldn't")
all_stopwords.remove("didn")
all_stopwords.remove("didn't")
all_stopwords.remove("doesn")
all_stopwords.remove("doesn't")
all_stopwords.remove("hadn")
all_stopwords.remove("hadn't")
all_stopwords.remove("hasn")
all_stopwords.remove("hasn't")
all_stopwords.remove("haven")
all_stopwords.remove("haven't")
all_stopwords.remove("isn")
all_stopwords.remove("isn't")
all_stopwords.remove("mightn")
all_stopwords.remove("mightn't")
all_stopwords.remove("mustn")
all_stopwords.remove("mustn't")
all_stopwords.remove("needn")
all_stopwords.remove("needn't")
all_stopwords.remove("shan")
all_stopwords.remove("shan't")
all_stopwords.remove("shouldn")
all_stopwords.remove("shouldn't")
all_stopwords.remove("wasn")
all_stopwords.remove("wasn't")
all_stopwords.remove("weren")
all_stopwords.remove("weren't")
all_stopwords.remove("won")
all_stopwords.remove("won't")
all_stopwords.remove("wouldn")
all_stopwords.remove("wouldn't")
all_stopwords.append('wow')
print('all_stopwords: ')
print(all_stopwords)
all_stop_words_set = set(all_stopwords)

print('----------------------------------------------')
print('Cleaning the texts')
corpus = []
print('dataset.shape[0]')
print(dataset.shape[0])

for i in range(0, dataset.shape[0]): # dataset.shape[0] -> num of rows - in this case 1000

  #replace anything that is not alphanumeric with spaces
  #  use the Review column, at row 'i'
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])

  review = review.lower()
  review_split = review.split()

  # print('review_split:')
  # print(review_split)
  # exit()
  
  # add to review_split all words that are not included at the stop-words
  #  also, stem each word (see PorterStemmer above)
  review_split = [ ps.stem(word) #stem this word
                  for word in review_split #word from word-array list
                  if not word in all_stop_words_set #if this word is not listed at the stop_words
                ]
  
  review = ' '.join(review_split)
  corpus.append(review)

# end of for i in range(0, dataset.shape[0]):
#--------------

print('----')
print('\ncorpus first 20 rows')
for i in corpus[0:20]:
    print(f'    {i}')
print('----')


print('----------------------------------------------')
print('Creating the Bag of Words model')

from sklearn.feature_extraction.text import CountVectorizer

print('checking how many words we have from all reviews')
print('   you can disable this step, this is a demonstration only')
count_vectorizer = CountVectorizer()
x = count_vectorizer.fit_transform(corpus).toarray()
print('corpus shape')
print(len(corpus))
print('x shape')
print(x.shape)
print(f'we have {x.shape[1]} unique words')
print('now, let\'s do for real with 1500 words ')
print('----------------------------------------------')

#note changing the max_features seems to have random effect under 500, and then onwards the effect
#  seems to be stable not improving much
count_vectorizer = CountVectorizer(max_features = 1500)
x = count_vectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values #all rows, only the last column - we don't transform because this is the answer we are looking for

print('x shape')
print(x.shape)
print('x:')
print(x)
print('----')
print('y shape')
print(y.shape)
print('y - first 20 elements')
print(y[0:20])


print('----------------------------------------------')
print('Splitting the dataset into the Training set and Test set')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

print('----------------------------------------------')
print('Training the Naive Bayes model on the Training set')
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)


print('----------------------------------------------')
print('Predicting the Test set results')
y_pred = classifier.predict(x_test)

print('x_test')
print(x_test)
print('----')
print('y_test')
print(y_pred)


print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))



print('----------------------------------------------')
print('Making the Confusion Matrix')
from sklearn.metrics import confusion_matrix, accuracy_score
cm_result = confusion_matrix(y_test, y_pred)
print(cm_result)
accuracy_score(y_test, y_pred)


accuracy_score_result = accuracy_score(y_test, y_pred)

print('----')
print(f'    True Negatives: {cm_result[0][0]} - False Negatives: {cm_result[1][0]}')
print(f'    True Positives: {cm_result[1][1]} - False Positives: {cm_result[0][1]}')
print('----')
print('Accuracy Score:')
print(accuracy_score_result)