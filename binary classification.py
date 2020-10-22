# 1. Load Libraries
import os
import time
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# define dataset, output, model, vectorizer folder names

dataset_path = 'dataset'
output_path = 'output'
model_path = 'model'
vectorizer_path = 'vectorizer'

# create directories
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(vectorizer_path):
    os.makedirs(vectorizer_path)

# Data Preprocessing
def preprocessing(X):
    documents = []
    stemmer = WordNetLemmatizer()

    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)
    return documents

# predict 
def predict(classifier, X_test):
    y_pred = classifier.predict(X_test)
    return y_pred

# inverse transform
def inverse_transform(encoder, y_pred):
    label_pred = encoder.inverse_transform(y_pred)
    return label_pred

# 2. load dataset
dataframe = pd.read_csv(os.path.join(dataset_path, "training_data_full.csv"))
dataset = dataframe.values

# 3. Feature Engineering
# Generate X, Y data
X = dataset[:, 0]
Y = dataset[:, 1]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

print("---Start training---")
# get time
start_time = time.time()

# Generate TFIDF feature values with removing stopwords
tfidfconverter = TfidfVectorizer(max_features=1000, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(preprocessing(X)).toarray()


# Dump the file
pickle.dump(tfidfconverter, open(os.path.join(vectorizer_path, "tfidfconverter.pkl"), "wb"))
print(X.shape)

# load the vetorizer from a pickle file
# this will be used in the future when you load the trained model
with open(os.path.join(vectorizer_path, "tfidfconverter.pkl"), 'rb') as vectorizer:
    tfidfconverter = pickle.load(vectorizer)

# 4. Training and Testing Sets
# test data size is 20 % of dataset
X_train, X_test, y_train, y_test = train_test_split(X, encoded_Y, test_size=0.2, random_state=0)

# create RF classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
# fit to train the classifier
classifier.fit(X_train, y_train)
# display training time
print('Training times : ', time.time() - start_time)
# 5. Evaluating the Model

# predict
y_pred = predict(classifier, X_test)

# display measurements of accuracies
print("---Display Accuracy---")
# confusion matrix
print("Confusion Matrix")
print(confusion_matrix(y_test,y_pred))
# accuracy report
print("Classification Report")
print(classification_report(y_test,y_pred))
# accracy score
print("Accuracy")
print(accuracy_score(y_test, y_pred))


# save the model into a pickle file
with open(os.path.join(model_path, 'binary_classifier.pkl'), 'wb') as picklefile:
    pickle.dump(classifier,picklefile)

# load the model from a pickle file
with open(os.path.join(model_path, 'binary_classifier.pkl'), 'rb') as training_model:
    classifier = pickle.load(training_model)

# 6. Predict the new unlabeled Dataset
print('---Start Predicting ')
# get time
start_time = time.time()

# read unlabeled data
test_dataframe = pd.read_csv(os.path.join(dataset_path, "test_data_10000.csv"))
# fetch the dataset
test_dataset = test_dataframe.values

# Get X data and fit it with TF-IDF Vectorizer
X_test_data = test_dataset[:, 0]
#  transform new data with fitted TF-IDF transformer
X_test_data = tfidfconverter.transform(preprocessing(X_test_data)).toarray()

# predict for new dataset
y_pred = predict(classifier, X_test_data)
# display Predicting time
print('Predicting times : ', time.time() - start_time)

# get decoded label(True, False)
label_pred = inverse_transform(encoder, y_pred)

# write the dataframe into csv file
test_dataframe['label'] = label_pred
test_dataframe.to_csv(os.path.join(output_path, "test_data_10000_output.csv"), index=False)
