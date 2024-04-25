import pandas as pd 
import numpy as np 
import re 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# from nltk import WordNetLemmatizer
from gensim.models import Word2Vec
from keras.models import load_model
model = load_model('notebook/model.h5')

stemmer = PorterStemmer()

def predict(input):
    text = re.sub('[^a-zA-Z]', ' ', input)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords.words('english')]
    text = ' '.join(text)
    word_tokenise =text.lower().split()
    wordmodel = Word2Vec.load("notebook/word2vec_model.model")
    X = np.array(np.mean([wordmodel.wv[word] for word in text if word in wordmodel.wv] or [np.zeros(wordmodel.vector_size)], axis=0))
    X = np.expand_dims(X, axis=0)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    output = model.predict(X)

    if(output>0.5):
        ans = 'Fake'
    else:
        ans = 'Not Fake'
    return ans 






