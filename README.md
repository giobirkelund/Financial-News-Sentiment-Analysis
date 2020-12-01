# Financial-News-Sentiment-Analysis
## Description
Training a naive bayes ML model with financial news articles to determine whether an unseen article's sentiment is positive or negative.
Each feature in training consists of the title + first two sentences of the article. 
I performed multiple pre-processing techniques to remove noise, and extract the useful words and strings of words which are most useful to determine the sentiment.

The model was deployed with Flask as a REST api, providing the ability to test the model through an endpoint with a new stock news article URL, which will then return the predicted sentiment (pos, neg, or neutral). 

[views.py](https://github.com/giobirkelund/Financial-News-Sentiment-Analysis/blob/master/api/views.py) has the endpoint used to make a prediction with a new stock news article 
[loadData.py](https://github.com/giobirkelund/Financial-News-Sentiment-Analysis/blob/master/api/loadData.py) contains code for preprocessing, vectorizing, training, and evaluating the ML model. 
