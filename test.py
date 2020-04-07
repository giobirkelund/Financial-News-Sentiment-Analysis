from newspaper import Article
from nltk.tokenize import sent_tokenize

url = 'https://finance.yahoo.com/news/stock-market-news-live-updates-april-7-221506245.html'
article = Article(url)
article.download()

article.parse()

print(article.title + "\n")
sentences = sent_tokenize(article.text)
print(sentences[0] + " " + sentences[1])

