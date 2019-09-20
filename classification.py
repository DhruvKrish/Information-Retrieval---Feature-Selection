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

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def Classification(classifier, X_train, X_test, y_train, y_test, percent=0):
	print "\n--------%s:%d----------\n"%(classifier, percent)
	start1 = time.time()
	if classifier=='svm':	
		clf = svm.SVC(kernel='linear', C=1)
	elif classifier=='knn':
		clf= KNeighborsClassifier(n_neighbors=5)
	elif classifier=='nb':
		clf = GaussianNB()
	else:
		raise Exception("INVALID CLASSIFIER")
	clf.fit(X_train, y_train)
	print "Training time taken %f \n" % (time.time() - start1)

	start2 = time.time()
	y_pred = clf.predict(X_test).tolist()
	print "Testing time taken %f \n" % (time.time() - start2)
	print "Classification report:\n"
	print metrics.classification_report(y_test, y_pred)
	print "Accuracy: %f"% accuracy_score( y_test, y_pred )

