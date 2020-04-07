import nltk
from nltk import NaiveBayesClassifier
from nltk import classify
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords # denne brukes for å slette stoppord fra tekstene
import string # denne importeres for å kunne slette tegnsetting fra tekstene
from nltk import bigrams
from nltk import trigrams
from nltk.stem import PorterStemmer
import re
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# api key: 6onk1z6389zttkq1luwvdtlvaffre9orbymgtckp
import pandas as pd
import numpy as np
import json
import random
from pandas.io.json import json_normalize
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from sklearn.model_selection import train_test_split

with open('stocknews.json', 'r') as f:
    datastore = json.load(f,strict=False)
#Use the new datastore datastructure
dataset = json_normalize(datastore['data'])[['title','text','sentiment']]
print(len(dataset.index))
dataset.sort_values("title", inplace = True) 
# dropping ALL duplicte values 
dataset.drop_duplicates(subset ="title", 
                     keep = False, inplace = True) 
dataset = sklearn.utils.shuffle(dataset)
# print(len(dataset.index))
# print("Number of Positive:",list(dataset["sentiment"]).count("Positive"))
# print("Number of Negative:",list(dataset["sentiment"]).count("Negative"))
# print("Number of Neutral:",list(dataset["sentiment"]).count("Neutral"))


#Process

# def removeEntities(wordsWithEntities):
#     spacy.prefer_gpu()
#     nlp = spacy.load("en_core_web_sm")
#     document = nlp(wordsWithEntities)
#     cleaned = []
#     for token in document:
#         if not token.ent_type:
#             cleaned.append(token.text.strip()) 
#     return cleaned
#make all words go to root word to reduce clutter feks connection, connected, connecting become connect

def stemWords(unstemmed):
    ps = PorterStemmer()
    stemmed_words=[]
    for w in unstemmed.split():
        stemmed_words.append(ps.stem(w))
    return stemmed_words

def preprocessing(dataset):
    eng_stopwords = set(stopwords.words('english')) - set(['isn','not','against','didnt','don','above','below','up','down','under','couldn','didn','doesn','hadn','hasn','haven','isn'])
    dataset['cleaned'] = None #creating new column for cleaned up text
    for index, row in dataset.iterrows():
        
        words = re.sub("[^(\w)]", " ", str(row['title'])+" "+str(row['text']))
        words = stemWords(words)
        # words = removeEntities(words)
        # Cleaning of list "words" and Inserting into pdf column
        # x.lower() for x in words if x.lower() not in eng_stopwords and x not in string.punctuation and not x.isdigit() and len(x)>2]
        cleaned = [x.lower() for x in words if x.lower() not in eng_stopwords and x not in (string.punctuation+")"+"(") and not x.isdigit()]
        cleaned =" ".join(cleaned)
        dataset.at[index, 'cleaned'] = cleaned
        # dataset.insert(index, 'cleaned', cleaned, allow_duplicates=False)

def testNewArticle(url):
    from newspaper import Article
    from nltk.tokenize import sent_tokenize

    article = Article(url)
    article.download()
    article.parse()
    title = article.title
    sentences = sent_tokenize(article.text)
    return title,(sentences[0] + " " + sentences[1])



def main():
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegressionCV      # using multiNomial Naive Bayes as classifier
    from sklearn.ensemble import RandomForestClassifier

    
    preprocessing(dataset)
    print(dataset)
    # eng_stopwords = set(stopwords.words('english')) - set(['isn','not','against','didnt','don','above','below','up','down','under','couldn','didn','doesn','hadn','hasn','haven','isn'])
    


    vectorizer = TfidfVectorizer (max_df=0.9, ngram_range=(1,2))
    processed_features = vectorizer.fit_transform(dataset['cleaned']).toarray()
    # print(processed_features)
    X_train, X_test, Y_train, Y_test = train_test_split(processed_features, dataset['sentiment'], train_size = 0.8 ,test_size=0.2, random_state=42)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    # text_classifier = SVC(kernel='linear')

    text_classifier = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    # text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    text_classifier.fit(X_train, Y_train)
    
    predictions = text_classifier.predict(X_test)
    
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    print(confusion_matrix(Y_test,predictions))
    print(classification_report(Y_test,predictions))
    print(accuracy_score(Y_test, predictions))
            #"title": "Opinion: There's no doubt we're in a recession",
            #"text": "On Friday, the Bureau of Labor Statistics announced that the unemployment rate rose to 4.4% in March, up from 3.5% in February. In addition, it reported 701,000 jobs were lost.",
            #negative

            # "title": "February JOLTS numbers beat expectations at 6.9 million",
            # "text": "CNBC's Rick Santelli reports on the latest JOLTS numbers from February, before the mass shutdowns across the country.",
            # "source_name": "CNBC Television",
            # #positive


    # tests = {
    #     'title': ["Opinion: There's no doubt we're in a recession"],
    #     'text': ["On Friday, the Bureau of Labor Statistics announced that the unemployment rate rose to 4.4% in March, up from 3.5% in February. In addition, it reported 701,000 jobs were lost."],
    #     'sentiment': ["Negative"]
    #     }

    tests = pd.DataFrame(columns = ('title','text'))

    url = input("Paste in an articles url to see it's sentiment prediction: ")
    newtitle,newtext = testNewArticle(url)
    
    tests.loc[0] = [newtitle,newtext]
    
    
    preprocessing(tests)
    # print(tests)
    test = tests['cleaned']
    prediction = text_classifier.predict(vectorizer.transform(test))
    tests['prediction'] = prediction
    print(tests)

    # vectorizer.fit_transform(test)
    
    # text_classifier.predict(test)
    # pred = text_classifier.predict(test)
    # y_pred = clf.predict(docs_test_tfidf)
    # print(sklearn.metrics.accuracy_score(Y_test, y_pred))


main()




