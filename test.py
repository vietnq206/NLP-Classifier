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
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# %%

ENGLISH_STOP_WORDS = ['my', 'between', 'others', 'more', 'thence', 'none', 'these', 'was', 'which', 'most', 'forty',
                      'do', 'hasnt', 'for', 'found', 'fifteen', 'anything', 'became', 'fifty', 'being', 'someone',
                      'sincere', 'latterly', 'somewhere', 'describe', 'up', 'whereas', 'become', 'over', 'system',
                      'there', 'perhaps', 'only', 'go', 'to', 'yet', 'part', 'the', 'amongst', 'nine', 'hereupon', 'of',
                      'against', 'under', 'every', 'first', 'however', 'why', 'such', 'nowhere', 'give', 'as', 'around',
                      'will', 'since', 'name', 'thick', 'yours', 'already', 'see', 'it', 'detail', 'so', 'then',
                      'anywhere', 'seeming', 'thru', 'behind', 'eight', 'am', 'both', 'whereby', 'hereafter', 'further',
                      'here', 'must', 'five', 'any', 'moreover', 'after', 'etc', 'also', 'cannot', 'some', 'one',
                      'once', 'across', 'where', 'formerly', 'well', 'show', 'via', 'from', 'themselves', 'yourselves',
                      'co', 'nevertheless', 'that', 'next', 'many', 'front', 'due', 'own', 'therefore', 'could', 'very',
                      'same', 'else', 'several', 'beside', 'alone', 'whenever', 'latter', 'in', 'without', 'because',
                      'anyone', 'wherein', 'anyhow', 'mill', 'can', 'fire', 'thus', 'though', 'everyone', 'least',
                      'side', 'how', 'into', 'before', 'thin', 'hundred', 'neither', 'has', 'himself', 'never',
                      'together', 'inc', 'thereupon', 'a', 'top', 'is', 'noone', 'done', 'fill', 'everywhere', 'onto',
                      'nobody', 'within', 'always', 'please', 'are', 'at', 'other', 'seemed', 'no', 'an', 'had',
                      'another', 'amount', 'below', 'he', 'mine', 'along', 'those', 'ourselves', 'each', 'should',
                      'thereafter', 'interest', 'not', 'this', 'all', 'rather', 'down', 'have', 'herself', 'sometimes',
                      'afterwards', 'cry', 'four', 'their', 'upon', 'amoungst', 'namely', 'serious', 'whole', 'twelve',
                      'above', 'meanwhile', 'eg', 'her', 'mostly', 'through', 'un', 'whereupon', 'six', 'find',
                      'elsewhere', 'beforehand', 'them', 'been', 'full', 'his', 'everything', 'among', 'toward', 'put',
                      'nor', 'besides', 'even', 'back', 'be', 'we', 'whereafter', 'myself', 'per', 'its', 'whether',
                      'eleven', 'empty', 'herein', 'de', 'couldnt', 'who', 'hers', 'too', 'whatever', 'and', 'con',
                      'yourself', 'whose', 'ie', 'indeed', 'throughout', 'might', 'ltd', 'your', 'wherever', 'take',
                      'sixty', 'sometime', 'she', 'until', 'ours', 'him', 'during', 'thereby', 'otherwise', 'again',
                      'were', 'if', 'whither', 'cant', 'three', 'i', 'hence', 'when', 're', 'on', 'seems', 'us',
                      'whoever', 'hereby', 'our', 'you', 'often', 'twenty', 'off', 'two', 'but', 'few', 'although',
                      'bottom', 'out', 'about', 'than', 'still', 'or', 'becoming', 'towards', 'last', 'whence',
                      'enough', 'much', 'may', 'itself', 'less', 'almost', 'get', 'except', 'therein', 'third', 'bill',
                      'me', 'something', 'made', 'ever', 'what', 'while', 'former', 'anyway', 'becomes', 'by', 'beyond',
                      'keep', 'nothing', 'move', 'with', 'now', 'seem', 'would', 'call', 'ten', 'they', 'either',
                      'somehow', 'whom']

# %%

path_dataset = 'data_set'


# %%

# Preprocessing step
def basic_processing(text):
    text = re.sub('-{2,}', '', text)
    patURL = r"(?:http://|www.)[^\"]+"
    text = re.sub(patURL, 'website', text)
    text = re.sub('\.+', '.', text)
    text = re.sub('\\s+', ' ', text)
    return text


def stemSentence(sentence):
    porter = PorterStemmer()
    token_words = word_tokenize(sentence)
    token_words
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


# remove stopword
def remove_stopwords(text):
    stop_words = list(ENGLISH_STOP_WORDS)
    tokens = text.split(" ")
    result = [i for i in tokens if not i in stop_words]
    return " ".join(result)


def clean_doc(text):
    text = stemSentence(text)
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
        text = text.replace(punc, ' ')
    text = re.sub('\\s+', ' ', text)

    return text


# %%

# Function : read dataset from the folder
def read_data(folder_path):
    documents = []
    labels = []
    for category in os.listdir(folder_path):
        print("Label: ", category)
        path_new = folder_path + "/" + category + "/*.txt"
        for filename in glob.glob(path_new):
            with open(filename, 'r', encoding="utf-8") as file:
                try:
                    content = file.read()
                    documents.append(content)
                    labels.append(category)
                except:
                    print(filename)
    return documents, labels


X_data, y_data = read_data(path_dataset)

# %%

# Show the size of dataset
print(len(X_data), len(y_data))

# %%

# Show an example
print(X_data[0])
print(y_data[0])

# %%

# Apply preprocessing in whole dataset
X_data_preprocess = []
for index, data in enumerate(X_data):
    X_data_preprocess.append(clean_doc(data))

# show one example
print(X_data_preprocess[0])

# %%


# %%

# ngram level tf-idf
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 2))
X_train_tfidf = vectorizer.fit_transform(X_data_preprocess)

cv = KFold(n_splits=5, random_state=42, shuffle=True)


tuned_parameters={ 'C': [0.1, 1],
   'gamma': [1, 0.1],
   'kernel': ['rbf','linear','sigmoid']  }
scores = ['precision', 'recall']
# SVM model
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(X_train_tfidf, y_data)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
