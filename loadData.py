
# api key: 6onk1z6389zttkq1luwvdtlvaffre9orbymgtckp
import pandas as pd
import json
from pandas.io.json import json_normalize

with open('stocknews.json', 'r') as f:
    datastore = json.load(f,strict=False)

#Use the new datastore datastructure
dataset = json_normalize(datastore['data'])[['text','sentiment']]
# print(dataset)
# print("Number of Positive:",list(dataset["sentiment"]).count("Positive"))
# print("Number of Negative:",list(dataset["sentiment"]).count("Negative"))
# print("Number of Neutral:",list(dataset["sentiment"]).count("Neutral"))

#select the nth row, and select a column
# print(dataset.iloc[1]['text'])

# print(dataset[])


#Process
from nltk import NaiveBayesClassifier
from nltk import classify
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords # denne brukes for å slette stoppord fra tekstene
import string # denne importeres for å kunne slette tegnsetting fra tekstene
from nltk import bigrams
from nltk import trigrams
import re





def preprocessing(dataset, pos_articles, neg_articles,neut_articles):
    eng_stopwords = stopwords.words('english')
    dataset['cleaned'] = None #creating new column for cleaned up text
    for index, row in dataset.iterrows():
        # print(index, row['text'],row['sentiment'])
        words = re.sub("[^(\w)]", " ", str(row['text'])).split()
        # Cleaning of list "words" and Inserting into pdf column
        cleaned = [x.lower() for x in words if x.lower() not in eng_stopwords and x not in string.punctuation and len(x)>3 and not x.isdigit()]
        dataset.at[index, 'cleaned'] = cleaned
        # dataset.insert(index, 'cleaned', cleaned, allow_duplicates=False)
        
        if(row['sentiment'] == 'Negative'):                        
            neg_articles.append(cleaned)            
        elif(row['sentiment'] == 'Positive'):
            pos_articles.append(cleaned)
        elif(row['sentiment'] == 'Neutral'):
            neut_articles.append(cleaned)
    return neg_articles, pos_articles, neut_articles

def makeFeatures(cleanedWords):
    cleaned = list(cleanedWords)
    for bi in bigrams(cleanedWords):
        cleaned.append(bi)
    # #using trigrams
    # for tri in trigrams(cleanedWords):
    #     cleaned.append(tri)

    words_dict = dict([word, True] for word in cleaned)
    return words_dict

def splitData(neg_articles, pos_articles, neut_articles):
    #proportionalize for our dataset 60/20/20


    trainpos = int(len(pos_articles)*(3/5))
    devpos = int(len(pos_articles)*(1/5))

    trainneg = int(len(neg_articles)*(3/5))
    devneg = int(len(neg_articles)*(1/5))

    trainneut = int(len(neut_articles)*(3/5))
    devneut = int(len(neut_articles)*(1/5))


    #scalable division into train dev test.
    #60 train, 20 dev, 20 test.
    train_set = neg_articles[0: trainneg] + pos_articles[0: trainpos] + neut_articles[0: trainneut]
    dev_set = neg_articles[trainneg:devneg+trainneg] + pos_articles[trainpos:devpos+trainpos] + neut_articles[trainneut: devneut+trainneut]
    test_set = neg_articles[devneg+trainneg:devneg+trainneg+devneg] + pos_articles[devpos+trainpos:devpos+trainpos+devpos] + neut_articles[devneut+trainneut:devneut+trainneut+devneut]

    return train_set,dev_set,test_set


def main():
    neg_articles= []
    pos_articles= []
    neut_articles= []

    neg_articles, pos_articles, neut_articles = preprocessing(dataset, pos_articles, neg_articles,neut_articles)
    
    # print(pos_articles)
    print(dataset)

    neg_articles_feat= []
    for words in neg_articles:
        neg_articles_feat.append((makeFeatures(words),'neg'))
    pos_articles_feat= []
    for words in pos_articles:
        pos_articles_feat.append((makeFeatures(words),'pos'))
    neut_articles_feat= []
    for words in neut_articles:
        neut_articles_feat.append((makeFeatures(words),'neut'))

    trainSet,devSet,testSet = splitData(neg_articles_feat, pos_articles_feat, neut_articles_feat)
    print(len(trainSet),len(testSet),len(devSet))
    classifier = NaiveBayesClassifier.train(trainSet)

    accuracy = classify.accuracy(classifier, devSet)

    print("Accuracy on dev_set: %0.3f" % accuracy)
    

    print (classifier.show_most_informative_features(10))
    accuracy = classify.accuracy(classifier, testSet)
    print("Accuracy on our test set: %0.3f" % accuracy)


main()




