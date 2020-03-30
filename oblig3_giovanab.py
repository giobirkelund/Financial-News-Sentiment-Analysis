"""
1)
    1.1)
    "Per slår Ola med boka"
    En tolnkningen er at Per slår Ola med boka
    en annen er at Per slår Ola som leser en bok

    1.2)
    S -> NP PP
    NP -> NP VP
    We change the rules now so that the original sentence can be interpreted that it is Per
    who hits Ola using the book as a weapon. 
    We make a rule that the sentence can be NP and PP then that allows 
    med boka to be: 

    PP -> P(med) + NP(boka).
    and Per Slår Ola is 
    NP -> (NP(Per) + VP (VP(slår)+ NP(Ola)).

    The sentence is now parsed in a way that it would be interpreted differently.

    1.3)
    VP -> VP PP is a recursive rule because VP is made up of a vp, which means you can 
    keep adding on phrases which extend the sentence, as long as there is a new PP. The new PP can be combined with the
    previous VP to make a new VP, and so on.  
    
"""

#2
"""
notes:
pre-modifiers
determiners: the house
quantifiers: few houses
numbers: 5 houses
adjectives: blue house

"""
import nltk
from nltk.corpus import conll2000
from nltk import NaiveBayesClassifier
from nltk import classify
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords # denne brukes for å slette stoppord fra tekstene
import string # denne importeres for å kunne slette tegnsetting fra tekstene
from nltk import bigrams
from nltk import trigrams

grammar = '''
NP: 
    {<DT|JJ|NN.*>+} #Chunk sequences of DT, JJ, NN
    {<NN|NNP>+} #one or more substantives or pronoun 
    {<1-9>*<NN>} # one or more numbers followed by a noun
    {<NN><IN>} # noun followed by a preposition/subordinating conjunction
    {<NN><NNS>} # noun followed by a plural noun 
 
'''

cp = nltk.RegexpParser(grammar)

training_chunks = conll2000.chunked_sents("train.txt", chunk_types=["NP"])
# print(training_chunks)
print(cp.evaluate(training_chunks))

"""
3.
    1. The Naive Bayes formula is derived from the original probability P(k|v) becuase 
    the equation finds the probability of k given v. We use the bayes rule to find this. 
    Go from: Probability: P(A|B) = P(A, B)/P(B)

    To: Product: P(A, B) = P(A|B)P(B) = P(B|A)P(A)
    
    P(A|B) =  P(B|A)P(A)/P(B) - Rearrange in order to get Bayes theorem.
    
    This is relevant in the naive bayes formula
    because it finds a class k given v. Essentially, this is used to find the probability of each word in a given class. and multiply them all together.
    Then, we use that data to find the probability of the entire class by using argmax to find the bigger value of the two probabilities.
    
    
    2.  prior sansynnlighet : Pos = 3/7 Neg = 4/7
    
        num of word given class + 1
        ----------------------------
        numwords in class + unique words

        førsteklasses artist men dårlig og kjedelig album
        4/7*
        P(førsteklasses|-) = (0 + 1)/(20 + 30) = 1/50
        P(artist|-) = (0 + 1)/(20 + 30) = 1/50
        P(men|-) = (0 + 1)/(20 + 30) = 1/50
        P(dårlig|-) = (1 + 1)/(20 + 30) = 2/50
        P(og|-) = (0 + 1)/(20 + 30) 1/50
        P(kjedelig|-) = (1 + 1)/(20 + 30) = 2/50
        P(album|-) = (1 + 1)/(20 + 30) = 2/50
        )
        4/7((1/50)(1/50)(1/50)(1/50)(2/50)(2/50)(1/50))
        - Sansynnlighet: 2.9257142857142856e-12
        

        3/7*
        P(førsteklasses|+) = (1 + 1)/(17 + 30) = 2/47
        P(artist|+) = (0 + 1)/(17 + 30) = 1/47
        P(men|+) = (0 + 1)/(17 + 30) = 1/47
        P(dårlig|+) = (1 + 1)/(17 + 30) = 2/47
        P(og|+) = (1 + 1)/(17 + 30) = 2/47
        P(kjedelig|+) = (0 + 1)/(17 + 30) = 1/47
        P(album|+) = (1 + 1)/(17 + 30) = 2/47
        
        (3/7)((1/47)(1/47)(1/47)(2/47)(2/47)(2/47)(2/47))
        + Sansynnlighet: 1.3534997871546315e-11

        The positive class has a higher probability because the positive probability > negative probability. 
        
        fortreffelig orkester og flott album
        4/7*
        P(fortreffelig|-) = (0 + 1)/(20 + 30) = 1/50
        P(orkester|-) = (1 + 1)/(20 + 30) = 2/50
        P(og|-) = (0 + 1)/(20 + 30) = 1/50
        P(flott|-) = (0 + 1)/(20 + 30) = 1/50
        P(album|-) = (1 + 1)/(20 + 30) = 2/50
        (4/7)((1/50)(1/50)(1/50)(2/50)(2/50))
        - Sansynnlighet: 7.314285

        3/7*
        P(fortreffelig|+) = (1 + 1)/(17 + 30)=2/47
        P(orkester|+) = (1 + 1)/(17 + 30)=2/47
        P(og|+) = (1 + 1)/(17 + 30)=2/47
        P(flott|+) = (0 + 1)/(17 + 30)= 1/47
        P(album|+) = (1 + 1)/(17 + 30)= 2/47
        3/7((2/47)(2/47)(2/47)(2/47)(1/47))
        + Sansynnlighet: 2.989881

        fortreffelig orkester og flott album
        fikk klassen Negativ fordi 7.31 > 2.99
        The positive class has a higher probability because the positive probability > negative probability. 

It is most probable that both sentences are classified as positive.

    3. I disagree with the first classification because although they say it's a first class artist,
    the review about the album is that it's actually boring and bad. Thus, the review should have actually been negative. 

        A solution for this would be to use more training data which will be able to recognise more positive words and negative words respectively.
        For example, the word "dårlig" was used in the positive class in one of the training examples. And because there's so few, it's heavily weighted as a positive word.
        Ultimately, the best solution would be to introduce context into the algorithm, as well as negation. If we have negation and context, the program will be able to realize that
        "ikke dårlig" is positive, instead of negative.
"""
#Oppg 4.1
print("OPPG 4.1")

#load the file
def load_corpus():
    corpus_root = 'NoReC/'
    reviews = PlaintextCorpusReader(corpus_root, '.*\.txt')
    return reviews

# Hente ut alle positive og negative ord som finnes i movie_reviews
def get_word_reviews(reviews, pos_reviews, neg_reviews):
    for fileid in reviews.fileids():
        if fileid.startswith('neg'):
            words = reviews.words(fileid)
            neg_reviews.append([x.lower() for x in words])
        elif fileid.startswith('pos'):
            words = reviews.words(fileid)
            pos_reviews.append([x.lower() for x in words])
    return pos_reviews, neg_reviews


# Bruk denne for å fordele dataen din i train, dev, og test set.
def splitdata(pos_reviews_feat, neg_reviews_feat):
    test_set = pos_reviews_feat[:122] + neg_reviews_feat[:122]
    dev_set = pos_reviews_feat[122:182] + neg_reviews_feat[122:182]
    train_set = pos_reviews_feat[182:] + neg_reviews_feat[182:]

    return test_set, dev_set, train_set

# Funksjon for å hente ut trekk til vår Naive Bayes.
def document_features(document, word_features): #tar inn et dokument
    document_words = set(document) # bruker sett for å enkelt kunne hente ut alle unike ord fra dokumentet
    features = {}
    for word in word_features: #sjekke om hvert ord i word_features finnes i dokumentet
        features['contains({})'.format(word)] = (word in document_words)
    return features

#-------------------
def main():
    pos_reviews = []
    neg_reviews = []
    reviews = load_corpus()

    pos_reviews, neg_reviews = get_word_reviews(reviews, pos_reviews, neg_reviews)
    stopwords_no = stopwords.words('norwegian')


    # Her velger vi de 1000 mest frekvente ord.
    all_words = nltk.FreqDist(w.lower() for w in reviews.words())
    word_features = list(all_words)[:1000]
    print("")

    # Du skal bruke fordelingen av dataen gitt i denne koden. Bruk derfor funksjonen splitdata()

    # lage trekk for positive anmeldelser

    documents = []

    for document in pos_reviews:
        documents.append((document, "pos")) #lage en liste av (text, klasse) par som skal brukes ved trening og testing

    for document in neg_reviews:
        documents.append((document, "neg")) #lage en liste av (text, klasse) par som skal brukes ved trening og testing
    # Du kan bruke listen av stoppord som finnes i NLTK, for å bruke den norske listen kan du bare skrive:
    # stopwords_no = stopwords.words('norwegian')
    featuresets = [(document_features(d, word_features), c) for (d,c) in documents] #bruke forrige funksjon til å generere trekk

    train_set, test_set = featuresets[100:], featuresets[:100] #dele dataen i train og test sets


    classifier = nltk.NaiveBayesClassifier.train(train_set)

    accuracy = classify.accuracy(classifier, test_set) #Regne ut accuracy på vår Naive Bayes

    print ("accuracy:", accuracy)
    print()
    print("10")
    print (classifier.show_most_informative_features(10))
    print("20")
    print (classifier.show_most_informative_features(20))
    print("30")
    print (classifier.show_most_informative_features(30))

"""

I think the results are decent, and can be true in some areas. However there are many secenarios 
where the words chosen as positive or negative without knowing how the word is used in the context.
For example, the word "rikt" was classified as positive. Though it was classified as such, it could
 be false in reality because the context was not taken into account. 
If a text had negation before "rikt" then it would be a negative thing. However, since this model only
 classifies words based on the most frequent 1000 words, context is not taken into account.

One way you could get better results is by using bigrams and ngrams in order to extract context.
Then, the words before or after a word can be included in the algorithm in order to not only compare individual words but strings of words that
usually appear together. 



"""

if __name__ == "__main__":
    main()

"""

Spørsmål 1: I think the results give a decent result, but it's not very practical. For example, this method can determine whether a word is generally
used positively or negatively, but is completely unrelated to the context. 
For example: when looking at the 20 most informative words, contains(helse) = True neg : pos = 3.6 : 1.0

"helse" was gategorized as negative, but it could be falsely categorized in reality because the context wasn't accounted for.

For example, the bigram "dårlig helse" would be negative, while "god helse" is positive. This most frequent 1000 words method would have no way of knowning the context.
I would suggest to use bigrams, and trigrams in order to gain context and therefore increase the accurary of the model when identifying wether a word is positive or negative.

Spørsmål 2:
I think that the first three options are good, because they begin to provide context and remove abstractions such as punctuation and stop words.
The last option could also be good, but it would be essential to have a good set of negative words so that most negative words will be recognised.
"""


print("OPPG 4.2")
#load the corpus
def load_corpus():
    corpus_root = 'NoReC/'
    reviews = PlaintextCorpusReader(corpus_root, '.*\.txt')
    return reviews

# Hente ut alle positive og negative ord som finnes i movie_reviews
def get_word_reviews(reviews, pos_reviews, neg_reviews):
    for fileid in reviews.fileids():
        if fileid.startswith('pos'):
            words = reviews.words(fileid)
            pos_reviews.append(words)
        elif fileid.startswith('neg'):
            words = reviews.words(fileid)
            neg_reviews.append(words)
    return pos_reviews, neg_reviews


stopwords_norwegian = stopwords.words('english')
#clean the words, and create bag of words, bigrams, and trigrams with stoppwords.
def bag_of_ngrams_no_stopwords_punct(words):
    words_cleaned = []

    lowered = [w.lower() for w in words if w not in stopwords_norwegian and w not in string.punctuation]
    # using bigrams
    for bi in bigrams(lowered):
        words_cleaned.append(bi)

    #using trigrams
    for tri in trigrams(lowered):
        words_cleaned.append(tri)

    #using bag of words
    for word in words:
        word = word.lower()
        if word not in stopwords_norwegian and word not in string.punctuation:
            words_cleaned.append(word)
    words_dict = dict([word, True] for word in words_cleaned)

    return words_dict

def main():
        
    # Hente ut alle stoppord for engelsk
    reviews = load_corpus()

    pos_reviews = []
    neg_reviews = []
    pos_reviews, neg_reviews = get_word_reviews(reviews, pos_reviews, neg_reviews)

    stopwords_no = stopwords.words("norwegian")
    # lage trekk for positie anmeldelser
   
    pos_reviews_feat = []
    for words in pos_reviews:

        pos_reviews_feat.append((bag_of_ngrams_no_stopwords_punct(words), 'pos'))                    
    # lage trekk for negative anmeldelser
    neg_reviews_feat = []
    for words in neg_reviews:

        neg_reviews_feat.append((bag_of_ngrams_no_stopwords_punct(words), 'neg'))
    # Fordele dataen vår i test_set, dev_set, og train_set
    test_set = pos_reviews_feat[:200] + neg_reviews_feat[:200]
    dev_set = pos_reviews_feat[200:300] + neg_reviews_feat[200:300]
    train_set = pos_reviews_feat[300:] + neg_reviews_feat[300:]

    print(len(test_set),  len(dev_set), len(train_set))
    # printer 400 200 1400

    # Nå skal vi trene vår Naive Bayes klassifisserer
    classifier = NaiveBayesClassifier.train(train_set)

    # Så enkelt var det!
    # Nå skal vi se hvor god den er ved å teste den på dev_set
    accuracy = classify.accuracy(classifier, dev_set)

    print("Accuracy on dev_set: %0.3f" % accuracy)
    

    # Vi kan se på de 10 mest informative trekk ved å gjøre følgende
    
    # print (classifier.show_most_informative_features(10))

    accuracy = classify.accuracy(classifier, test_set)
    print("Accuracy on our test set: %0.3f" % accuracy)
main()


"""
2.1: Bag-Of-Words (BOW): uten stoppord og tegnsetting
Accuracy on dev_set: 0.69
natur = True              pos : neg    =     10.3 : 1.0
              menneskets = True              pos : neg    =     10.3 : 1.0
                poetiske = True              pos : neg    =     10.3 : 1.0
                   veier = True              pos : neg    =      9.7 : 1.0
                 russisk = True              pos : neg    =      8.3 : 1.0
               strålende = True              pos : neg    =      8.3 : 1.0
                  island = True              pos : neg    =      8.3 : 1.0
                  vakker = True              pos : neg    =      8.2 : 1.0
                       ” = True              pos : neg    =      7.8 : 1.0
                    kjem = True              pos : neg    =      7.8 : 1.0

2.2. BOW + bigrams: uten stoppord og tegnsetting, kombinasjon av unigrams og bigrams.
Accuracy on dev_set: 0.705
    poetiske = True              pos : neg    =     10.3 : 1.0
                   natur = True              pos : neg    =     10.3 : 1.0
              menneskets = True              pos : neg    =     10.3 : 1.0
                   veier = True              pos : neg    =      9.7 : 1.0
                 russisk = True              pos : neg    =      8.3 : 1.0
                  island = True              pos : neg    =      8.3 : 1.0
               strålende = True              pos : neg    =      8.3 : 1.0
                  vakker = True              pos : neg    =      8.2 : 1.0
                    kjem = True              pos : neg    =      7.8 : 1.0
                       ” = True              pos : neg    =      7.8 : 1.0


2.3. BOW + bigrams + trigrams: uten stoppord og tegnsetting, kombinasjon an unigrams, bigrams, og trigrams.
Accuracy on dev_set: 0.700
                poetiske = True              pos : neg    =     10.3 : 1.0
                   natur = True              pos : neg    =     10.3 : 1.0
              menneskets = True              pos : neg    =     10.3 : 1.0
                   veier = True              pos : neg    =      9.7 : 1.0
                  island = True              pos : neg    =      8.3 : 1.0
                 russisk = True              pos : neg    =      8.3 : 1.0
               strålende = True              pos : neg    =      8.3 : 1.0
                  vakker = True              pos : neg    =      8.2 : 1.0
                    kjem = True              pos : neg    =      7.8 : 1.0
                       ” = True              pos : neg    =      7.8 : 1.0
The most informative features for each one were relatively similar, but having occasional bigrams or trigrams in the mix. These, however, did not show up in the top 10.

The model which did the best for the test set was BOW + bigrams according to my data with a 70.5% on the dev set when compared to the unigrams, bigrams, and trigrams which was only 70.0%. 

This is suprising to me because I assumed since trigrams includes the bigrams it would certainly give more accurate results - but clearly this isn't the case (as long as my solution is correct.) I think the best model is dependent on the corpus.

We could use negation in addition in order to by hand define words that are typically used negatively to ensure some consistency. This would surely increase our accuracy as long as our list of words for negation was good.

"""

