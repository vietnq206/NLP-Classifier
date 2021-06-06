# -*- coding: utf-8 -*-

import os, string, re
import glob
import numpy as np
from sklearn.metrics import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


ENGLISH_STOP_WORDS = ['my', 'between', 'others', 'more', 'thence', 'none', 'these', 'was', 'which', 'most', 'forty', 'do', 'hasnt', 'for', 'found', 'fifteen', 'anything', 'became', 'fifty', 'being', 'someone', 'sincere', 'latterly', 'somewhere', 'describe', 'up', 'whereas', 'become', 'over', 'system', 'there', 'perhaps', 'only', 'go', 'to', 'yet', 'part', 'the', 'amongst', 'nine', 'hereupon', 'of', 'against', 'under', 'every', 'first', 'however', 'why', 'such', 'nowhere', 'give', 'as', 'around', 'will', 'since', 'name', 'thick', 'yours', 'already', 'see', 'it', 'detail', 'so', 'then', 'anywhere', 'seeming', 'thru', 'behind', 'eight', 'am', 'both', 'whereby', 'hereafter', 'further', 'here', 'must', 'five', 'any', 'moreover', 'after', 'etc', 'also', 'cannot', 'some', 'one', 'once', 'across', 'where', 'formerly', 'well', 'show', 'via', 'from', 'themselves', 'yourselves', 'co', 'nevertheless', 'that', 'next', 'many', 'front', 'due', 'own', 'therefore', 'could', 'very', 'same', 'else', 'several', 'beside', 'alone', 'whenever', 'latter', 'in', 'without', 'because', 'anyone', 'wherein', 'anyhow', 'mill', 'can', 'fire', 'thus', 'though', 'everyone', 'least', 'side', 'how', 'into', 'before', 'thin', 'hundred', 'neither', 'has', 'himself', 'never', 'together', 'inc', 'thereupon', 'a', 'top', 'is', 'noone', 'done', 'fill', 'everywhere', 'onto', 'nobody', 'within', 'always', 'please', 'are', 'at', 'other', 'seemed', 'no', 'an', 'had', 'another', 'amount', 'below', 'he', 'mine', 'along', 'those', 'ourselves', 'each', 'should', 'thereafter', 'interest', 'not', 'this', 'all', 'rather', 'down', 'have', 'herself', 'sometimes', 'afterwards', 'cry', 'four', 'their', 'upon', 'amoungst', 'namely', 'serious', 'whole', 'twelve', 'above', 'meanwhile', 'eg', 'her', 'mostly', 'through', 'un', 'whereupon', 'six', 'find', 'elsewhere', 'beforehand', 'them', 'been', 'full', 'his', 'everything', 'among', 'toward', 'put', 'nor', 'besides', 'even', 'back', 'be', 'we', 'whereafter', 'myself', 'per', 'its', 'whether', 'eleven', 'empty', 'herein', 'de', 'couldnt', 'who', 'hers', 'too', 'whatever', 'and', 'con', 'yourself', 'whose', 'ie', 'indeed', 'throughout', 'might', 'ltd', 'your', 'wherever', 'take', 'sixty', 'sometime', 'she', 'until', 'ours', 'him', 'during', 'thereby', 'otherwise', 'again', 'were', 'if', 'whither', 'cant', 'three', 'i', 'hence', 'when', 're', 'on', 'seems', 'us', 'whoever', 'hereby', 'our', 'you', 'often', 'twenty', 'off', 'two', 'but', 'few', 'although', 'bottom', 'out', 'about', 'than', 'still', 'or', 'becoming', 'towards', 'last', 'whence', 'enough', 'much', 'may', 'itself', 'less', 'almost', 'get', 'except', 'therein', 'third', 'bill', 'me', 'something', 'made', 'ever', 'what', 'while', 'former', 'anyway', 'becomes', 'by', 'beyond', 'keep', 'nothing', 'move', 'with', 'now', 'seem', 'would', 'call', 'ten', 'they', 'either', 'somehow', 'whom']

# Read data from google drive
path_dataset ="data_set"

# Preprocessing step
def basic_processing(text):
    text = re.sub('-{2,}','',text)
    patURL = r"(?:http://|www.)[^\"]+"
    text = re.sub(patURL,'website',text)
    text = re.sub('\.+','.',text)
    text = re.sub('\\s+',' ',text)
    return text
# remove stopword
def remove_stopwords(text):
    stop_words = ENGLISH_STOP_WORDS
    tokens = text.split(" ")
    result = [i for i in tokens if not i in stop_words]
    return " ".join(result)

def clean_doc(text):
    # apply basic preprocessing
    text = basic_processing(text)
    # Remove stop word in the text
    text = remove_stopwords(text)
    # Lower case
    text = text.lower()
    # multiple spaces removal
    text = re.sub(r"\?", " \? ", text)
    # Remove number in the text
    text = re.sub(r"[0-9]+", " ", text)
    # Remove punctuation
    for punc in string.punctuation:
        text = text.replace(punc,' ')
    text = re.sub('\\s+',' ',text)
    return text

# Function : read dataset from the folder
def read_data(folder_path):
    documents = []
    labels = []
    for category in os.listdir(folder_path):
        print("Label: ", category)
        path_new = folder_path+ "/"+category + "/*.txt"
        for filename in glob.glob(path_new):
            with open(filename,'r',encoding="utf-8") as file:
              try:
                  content = file.read()
                  documents.append(content)
                  labels.append(category)
              except:
                  print(filename)
    return documents, labels

X_data, y_data = read_data(path_dataset)

# Show the size of dataset
print(len(X_data),len(y_data))

# Show an example
print(X_data[0])
print(y_data[0])

# Apply preprocessing in whole dataset
X_data_preprocess = []
for index,data in enumerate(X_data):
    X_data_preprocess.append(clean_doc(data))

# show one example
print(X_data_preprocess[0])

# ngram level tf-idf 
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2,2))
X_train_tfidf = vectorizer.fit_transform(X_data_preprocess)

cv = KFold(n_splits=5, random_state=42, shuffle=True)

# naive_bayes model 
from sklearn.naive_bayes import MultinomialNB
model_naivebayes = MultinomialNB()

# evaluate cross validation 5 folds of Naive Bayes model
scores = cross_val_score(model_naivebayes, X_train_tfidf, y_data, scoring='accuracy', cv=cv)
# report performance
print(scores)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# K-nearest neighbors model 
from sklearn.neighbors import KNeighborsClassifier

model_knn =  KNeighborsClassifier(n_neighbors=3)
# evaluate cross validation 5 folds of KNeighborsClassifier model
scores = cross_val_score(model_knn, X_train_tfidf, y_data, scoring='accuracy', cv=cv)
# report performance
print(scores)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# Decision Tree model 
from sklearn.tree import DecisionTreeClassifier

model_dt =  DecisionTreeClassifier(random_state=42)
# evaluate cross validation 5 folds of KNeighborsClassifier model
scores = cross_val_score(model_dt, X_train_tfidf, y_data, scoring='accuracy', cv=cv)
# report performance
print(scores)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# SVM model 
from sklearn.svm import SVC

model_svm =  SVC(kernel="linear",gamma='auto', C=1.0,probability=True)
# evaluate cross validation 5 folds of SVM model
scores = cross_val_score(model_svm, X_train_tfidf, y_data, scoring='accuracy', cv=cv)
# report performance
print(scores)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

from sklearn.ensemble import VotingClassifier

soft_ensemble = VotingClassifier(estimators=[('svm', model_svm), ('knn', model_knn), ('nb', model_naivebayes)], voting='soft')

# evaluate cross validation 5 folds of KNeighborsClassifier model
scores = cross_val_score(soft_ensemble, X_train_tfidf, y_data, scoring='accuracy', cv=cv)
# report performance
print(scores)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# Because soft_ensemble model has the highest score using cross-validation 5 fold. We will use this model to train data
soft_ensemble.fit(X_train_tfidf, y_data)

# Demo with the example in the dataset
input_demo = X_data[0]
input_clean = clean_doc(input_demo)
input_tfidf = vectorizer.transform([input_clean])
predict = model_knn.predict(input_tfidf)
###########################################
print("Input: \n ", input_demo)
print("*"*100)
print("Input Preprocessing: \n ", input_clean)
print("*"*100)
print("Model predict: ", predict[0])
print("True Label: ", y_data[0])

# Demo with the input text
input_demo = "LeBron James scored 29 points for the Lakers, who are the first defending champions to lose at this stage since the San Antonio Spurs in 2015. The result also marks the first time that James has lost in the first round of the play-offs. The Suns will now face the Denver Nuggets in the Western Conference semi-finals after they clinched a 4-2 series win over the Portland Trail Blazers."
input_clean = clean_doc(input_demo)
input_tfidf = vectorizer.transform([input_clean])
predict = model_knn.predict(input_tfidf)
###########################################
print("Input: \n ", input_demo)
print("*"*100)
print("Input Preprocessing: \n ", input_clean)
print("*"*100)
print("Model predict: ", predict[0])

# Demo with the file  text
path_file = ""
with open(path_file, "r", encoding="utf8") as file:
  input_demo = file.read()
input_clean = clean_doc(input_demo)
input_tfidf = vectorizer.transform([input_clean])
predict = model_knn.predict(input_tfidf)
###########################################
print("Input: \n ", input_demo)
print("*"*100)
print("Input Preprocessing: \n ", input_clean)
print("*"*100)
print("Model predict: ", predict[0])