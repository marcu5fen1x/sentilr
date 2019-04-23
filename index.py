# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

import os
import json
from sklearn.externals import joblib
import flask
from flask import Flask
import pandas as pd
import nltk

from flask import render_template,request
app = Flask(__name__)

import sys

import tarfile

import time

import pyprind

import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

import re

from nltk.stem.porter import PorterStemmer

import nltk

from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

def preprocessor(text):

    text = re.sub('<[^>]*>', '', text)

    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',

                           text)

    text = (re.sub('[\W]+', ' ', text.lower()) +

            ' '.join(emoticons).replace('-', ''))

    return text

def tokenizer(text):

    return text.split()


# The flask app for serving predictions
app = flask.Flask(__name__)
# @app.route('/ping', methods=['GET'])
# def ping():
#     # Check if the classifier was loaded correctly
#     try:
#         classifier
#         status = 200
#     except:
#         status = 400
#     return flask.Response(response= json.dumps(' '), status=status, mimetype='application/json' )

@app.route('/')
def my_form():
    return render_template('index.html')

# @app.route('/', methods=['POST'])
# def my_form_post():
#     text = request.form['text']
#     processed_text = text.upper()
#     return processed_

def train(l):
    df = pd.read_csv('C:\Users\Aveen\PycharmProjects\senti\movie_data.csv', encoding='utf-8')

    stop = stopwords.words('english')

    X_train = df.loc[:2500, 'review'].values
    count = CountVectorizer()
    df['review'] = df['review'].apply(preprocessor)
    nltk.download('stopwords')

    y_train = df.loc[:2500, 'sentiment'].values

    X_test = df.loc[2500:, 'review'].values

    y_test = df.loc[2500:, 'sentiment'].values

    tfidf = TfidfVectorizer(strip_accents=None,

                            lowercase=False,

                            preprocessor=None)

    param_grid = [{'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0]},
                  ]

    lr_tfidf = Pipeline([('vect', tfidf),

                         ('clf', LogisticRegression(random_state=0))])

    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,

                               scoring='accuracy',

                               cv=5,

                               verbose=1,

                               n_jobs=-1)
    gs_lr_tfidf.fit(X_train, y_train)

    print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)

    print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

    clf = gs_lr_tfidf.best_estimator_

    print('Test Accuracy: %.3f' % clf.score(X_test, y_test))
    # l=np.array(l)
    tfidf.fit_transform(X_train)

    #input_tokenized = tfidf.transform(["today was a great day"])
    predictions = clf.predict(["today was a great day"])
    print(type(predictions))
    return predictions

@app.route('/invocations', methods=['POST'])
def transformation():
    # Get input JSON data and convert it to a DF
    text = request.form['text']
    print(text)
    l = []
    l.append(text)
    print(l)
    predictions = train(l)
    # Tokenize data and predict


    # Transform predicted labels (0 and 1) to easier to understand Negative and Positive labels
    predictions = list(map(lambda x: 'Positive' if x == 1 else 'Negative', predictions))

    # Transform predictions to JSON
    result = {'output': []}
    list_out = []
    for label in predictions:
        row_format = {'label': label}
        list_out.append(row_format)
    result['output'] = list_out
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
