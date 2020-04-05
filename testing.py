import newspaper
from newspaper import Article
import subprocess 

        
url = "https://www.cnn.com/2020/03/20/business/amazon-warehouse-employees-coronavirus/index.html"

article = Article(url)
article.download()
article.parse()
text = article.text

subprocess.run("pbcopy", universal_newlines=True, input=text)