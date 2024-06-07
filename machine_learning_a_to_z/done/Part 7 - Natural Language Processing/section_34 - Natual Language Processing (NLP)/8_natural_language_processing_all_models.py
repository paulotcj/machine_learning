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

# print(f'Dataset first 5 rows')
# print(dataset[0:5])

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

all_stop_words_set = set(all_stopwords)


print('----------------------------------------------')
print('Cleaning the texts')
corpus = []


for i in range(0, dataset.shape[0]): # dataset.shape[0] -> num of rows - in this case 1000

    #replace anything that is not alphanumeric with spaces
    #  use the Review column, at row 'i'
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])

    review = review.lower()
    review_split = review.split()


  
    # add to review_split all words that are not included at the stop-words
    #  also, stem each word (see PorterStemmer above)
    review_split = [    ps.stem(word) #stem this word
                        for word in review_split #word from word-array list
                        if not word in all_stop_words_set #if this word is not listed at the stop_words
                    ]

    review = ' '.join(review_split)
    corpus.append(review)

# end of for i in range(0, dataset.shape[0]):
#--------------




print('----------------------------------------------')
print('Creating the Bag of Words model')

from sklearn.feature_extraction.text import CountVectorizer


count_vectorizer = CountVectorizer()
x = count_vectorizer.fit_transform(corpus).toarray()


print('----------------------------------------------')

#note changing the max_features seems to have random effect under 500, and then onwards the effect
#  seems to be stable not improving much
count_vectorizer = CountVectorizer(max_features = 1500)
x = count_vectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values #all rows, only the last column - we don't transform because this is the answer we are looking for




print('----------------------------------------------')
print('Splitting the dataset into the Training set and Test set')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

print('----------------------------------------------')
print('Training')
accuracy_scores : dict[list] = {}

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.svm import SVC #support vector classification
classifier_Kernel_SVC = SVC(kernel = 'rbf', random_state = 0)
classifier_Kernel_SVC.fit(x_train, y_train)
y_pred_Kernel_SVC = classifier_Kernel_SVC.predict(x_test)
accuracy_score_result_Kernel_SVC = accuracy_score(y_test, y_pred_Kernel_SVC)
accuracy_scores['Kernel_SVC'] = [accuracy_score_result_Kernel_SVC,y_pred_Kernel_SVC]

# from sklearn.svm import SVC
classifier_SVM = SVC(kernel = 'linear', random_state = 0)
classifier_SVM.fit(x_train, y_train)
y_pred_SVM = classifier_SVM.predict(x_test)
accuracy_score_result_SVM = accuracy_score(y_test, y_pred_SVM)
accuracy_scores['SVM'] = [accuracy_score_result_SVM,y_pred_SVM]


from sklearn.neighbors import KNeighborsClassifier
classifier_KN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_KN.fit(x_train, y_train)
y_pred_KN = classifier_KN.predict(x_test)
accuracy_score_result_KN = accuracy_score(y_test, y_pred_KN)
accuracy_scores['KNeighbors'] = [accuracy_score_result_KN,y_pred_KN]


from sklearn.tree import DecisionTreeClassifier
classifier_decision_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_decision_tree.fit(x_train, y_train) 
y_pred_decision_tree = classifier_decision_tree.predict(x_test)
accuracy_score_result_tree = accuracy_score(y_test, y_pred_decision_tree)
accuracy_scores['DecisionTree'] = [accuracy_score_result_tree,y_pred_decision_tree]


from sklearn.ensemble import RandomForestClassifier
classifier_random_forest = RandomForestClassifier(n_estimators = 40, criterion = 'entropy', random_state = 0)
classifier_random_forest.fit(x_train, y_train)
y_pred_random_forest = classifier_random_forest.predict(x_test)
accuracy_score_result_random_forest = accuracy_score(y_test, y_pred_random_forest)
accuracy_scores['RandomForest'] = [accuracy_score_result_random_forest,y_pred_random_forest]


from sklearn.linear_model import LogisticRegression
classifier_logistic_regression = LogisticRegression(random_state = 0) # random_state = 0 -> get always the same results
classifier_logistic_regression.fit(x_train, y_train)
y_pred_logistic_regression = classifier_logistic_regression.predict(x_test)
accuracy_score_result_logistic_regression = accuracy_score(y_test, y_pred_logistic_regression)
accuracy_scores['LogisticRegression'] = [accuracy_score_result_logistic_regression,y_pred_logistic_regression]

print('----------------------------------------------')
print('Current models\' accuracy')
for key, value in accuracy_scores.items():
    print(f'{key} - {value[0]}')

print('----------------------------------------------')
print('filtering only models with accuracy >= 75%')
accuracy_scores = {key: value for key, value in accuracy_scores.items() if value[0] >= 0.75}
#---
accuracy_scores = dict(sorted(accuracy_scores.items(), key=lambda item: item[1][0], reverse= True))

# num_to_keep = len(accuracy_scores) // 2
num_to_keep = round(len(accuracy_scores) * 0.3)

print(f'    num of models to keep: {num_to_keep}')

accuracy_scores = dict(list(accuracy_scores.items())[0:num_to_keep])

for key, value in accuracy_scores.items():
    print(f'{key} - {value[0]}')
#---


print('----------------------------------------------')
print('Use the models to vote')
y_pred_final : list[int] = [0] * len(y_test)
for i in range(len(y_pred_final)):
    temp_y : int = 0
    for key, value in accuracy_scores.items():
        temp_y = temp_y + value[1][i]
    temp_y = round(temp_y/len(accuracy_scores))

    y_pred_final[i] = temp_y


print('----------------------------------------------')
print('Final accuracy score:')
accuracy_score_result_VOTED = accuracy_score(y_test, y_pred_final)
print(accuracy_score_result_VOTED)





    


   
