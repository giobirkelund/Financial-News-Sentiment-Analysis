import nltk
from nltk.corpus import stopwords # denne brukes for å slette stoppord fra tekstene
import string # denne importeres for å kunne slette tegnsetting fra tekstene
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# api key: 6onk1z6389zttkq1luwvdtlvaffre9orbymgtckp
import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from sklearn.model_selection import train_test_split
import pickle

def loadDataset():
    with open('stocknews.json', 'r') as f:
        datastore = json.load(f,strict=False)
    dataset = pd.json_normalize(datastore['data'])[['title','text','sentiment']]
    dataset.sort_values("title", inplace = True) 
    # dropping ALL duplicte values 
    dataset.drop_duplicates(subset ="title", 
                        keep = False, inplace = True) 
    dataset = sklearn.utils.shuffle(dataset)
    # print(len(dataset.index))
    # print("Number of Positive:",list(dataset["sentiment"]).count("Positive"))
    # print("Number of Negative:",list(dataset["sentiment"]).count("Negative"))
    # print("Number of Neutral:",list(dataset["sentiment"]).count("Neutral")
    return dataset


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

####################
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
        cleaned = [x.lower() for x in words if x.lower() not in eng_stopwords and x not in (string.punctuation+")"+"(") and not x.isdigit()]
        cleaned =" ".join(cleaned)
        dataset.at[index, 'cleaned'] = cleaned
        # dataset.insert(index, 'cleaned', cleaned, allow_duplicates=False)

def getArticleData(url):
    from newspaper import Article
    from nltk.tokenize import sent_tokenize

    article = Article(url)
    article.download()
    article.parse()
    title = article.title
    sentences = sent_tokenize(article.text)
    return title,(sentences[0] + " " + sentences[1])



def trainModel(dataset):
    ##global variables
    vectorizer = TfidfVectorizer (min_df = 5,max_df = .8, ngram_range=(1,2))
    classifier = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    preprocessing(dataset)
    processed_features = vectorizer.fit_transform(dataset['cleaned']).toarray()
    X_train, X_test, Y_train, Y_test = train_test_split(processed_features, dataset['sentiment'], train_size = 0.8 ,test_size=0.2, random_state=42)
    classifier.fit(X_train, Y_train)
    
    #create a picle of the classifier to use it later.
    pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
    pickle.dump(classifier, open('preTrainedModel.pkl', 'wb'))
    pickle.dump(dataset, open('dataset.pkl', 'wb'))

    predictions = classifier.predict(X_test)
    
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    # print(confusion_matrix(Y_test,predictions))
    # print(classification_report(Y_test,predictions))
    # print(accuracy_score(Y_test, predictions))
    accuracy = float(np.round(accuracy_score(Y_test, predictions), 2))
    return accuracy


def testArticle(url):
    # processed_features = vectorizer.fit_transform(dataset['cleaned']).toarray()
    tests = pd.DataFrame(columns = ('title','text'))
    
    newtitle,newtext = getArticleData(url)
    
    tests.loc[0] = [newtitle,newtext]

    preprocessing(tests) #clean the test data
    # print(tests)
    test = tests['cleaned'] #put cleaned data in a new column

    #need to retrieve classifier and vectorizer which were pickled
    import os
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    modelPath = os.path.join(THIS_FOLDER, 'preTrainedModel.pkl')
    vectorizerPath = os.path.join(THIS_FOLDER, 'vectorizer.pkl')

    classifier = pickle.load(open(modelPath,'rb'))
    vectorizer = pickle.load(open(vectorizerPath,'rb'))
    prediction = classifier.predict(vectorizer.transform(test))
    tests['prediction'] = str(prediction)
    return newtitle,str(prediction)

def main():
# we need to load the dataset
    dataset = loadDataset()
    accuracy = trainModel(dataset)
    print(accuracy)
    url = input("Paste in an articles url to see it's sentiment prediction: ")
    title,prediction = testArticle(url)
    print(title)
    print(prediction)

if __name__ == "__main__":
    #Do some local work which should not be reflected while importing this file to another module.
    main()