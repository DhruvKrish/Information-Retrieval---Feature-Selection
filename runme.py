import shutil
import os
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import time
from collections import Counter
import math
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings("ignore")
#from stemming.porter2 import stem
from nltk.corpus import stopwords
import itertools
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
import string
import nltk
from collections import Counter
from scipy.stats import norm
import heapq
from scipy import sparse
from collections import Counter
from scipy.stats import norm
import cPickle as pickle
from sklearn import metrics

from sklearn import cross_validation
from sklearn.metrics import classification_report


from classification import Classification

'''
Place the Classic dataset (data files) on creating a folder by name "classic".
Create another folder "Classic_Dataset" and 4 sub folders by name "med", "cran", "cicsi", "cacm".
 
The hardcoded number(a, b, c, d) of files will be copied from the "classic" to corresponding folders ("cacm", "med", "cisi", "cran").

'''
# Move dataset files into the required format
print "Preparing dataset..."
from prepare_dataset import prepare_dataset
prepare_dataset()


#Load The files/dataset
cwd = os.getcwd()
load_path = cwd + "/Classic_Dataset"
dataset=load_files(load_path, description=None, categories=None, load_content=True, shuffle=False, encoding=None, decode_error='strict', random_state=0)

#Class names and assigned numbers
class_names= list(dataset.target_names)
class_num = len(class_names)
class_numbers = []
for i in range(0, len(class_names)):
	class_numbers.append(i)

#Document class labels
d_labels = dataset.target
#Data from the dataset
vdoc = dataset.data

#Stemming the words
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

#Tokenizing each word
def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems



print "Preparing doc-term vector..."
#Count Tokenizer --> Finding doc-term frequency matrix
vec = CountVectorizer(tokenizer=tokenize, stop_words='english')
data = vec.fit_transform(vdoc).toarray()


voc = vec.get_feature_names()

#Finding Vocabulary
vocabulary = voc
voc_num = len(vocabulary)
doc_num = data.shape[0]



#voc is total vocabulary
#data is doc * voc	
#class doc is class * doc
#cl_voc is vocabluray of each doc 
#class_voc is voc of each class

#final doc-term tfidf vector
tf_vec = TfidfTransformer(use_idf=True).fit(data)
vectors = tf_vec.transform(data)

class_doc = []
class_doc.append([i for i, j in enumerate(d_labels) if j == 0])
class_doc.append([i for i, j in enumerate(d_labels) if j == 1])
class_doc.append([i for i, j in enumerate(d_labels) if j == 2])
class_doc.append([i for i, j in enumerate(d_labels) if j == 3])

cl_voc = []
for item in data:
	cl_voc.append([i for i, j in enumerate(item) if j != 0])

class_voc = []
for item in class_doc:
	temp = [cl_voc[x] for x in item]
	class_voc.append(list(set(list(itertools.chain.from_iterable(temp)))))

class_vec = []
for i in range(0,len(class_doc)):
	s = []
	for item in class_doc[i]:
		s.append(vectors.getrow(item).toarray().tolist()[0])
	class_vec.append(csr_matrix(s))


########################
# OUR CODE STARTS HERE #
#######################


import my_feature_selection

# Do the feature selection for each of the required percentages
print "Doing feature ranking..."
percents = [10,20,40,50,70,80,100]
rank_start_time = time.time()
ranked_features = my_feature_selection.do_ranking( data, d_labels, 4 )
print "\t - Time taken to do feature ranking: %f"% (time.time()-rank_start_time)
print "Selecting top p features..."
selected_features = {}
for p in percents:
	print "\t- Selecting top %i" % p
	
	selected_features[p] = my_feature_selection.select_top_percent(ranked_features, voc, vectors,  p) #new_vectors.toarray()
y2 = d_labels

# Run all the classifiers for all the percentages of features selected.
print "Starting Classification..."
classifiers = [ 'svm', 'knn', 'nb']
for classifier in classifiers: 
	for p in percents: #selected_features:
		X2 = selected_features[p]
		X_train2, X_test2, y_train2, y_test2 = cross_validation.train_test_split(X2, y2, test_size=0.4, random_state=0)
		Classification(classifier, X_train2, X_test2, y_train2, y_test2, p)


print "Done!"
#Also find the ACCURACY VALUES

#THE END
