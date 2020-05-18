from flask import Blueprint, request
from . import db
from .models import Article
from .loadData import testArticle

main = Blueprint('main', __name__)

@main.route('/test_url', methods = ['POST'])
def test_url():
    link = request.get_json(force=True)
    title, prediction = testArticle(link['url'])
    output = "Title: "+title +"\nPrediction: "+ prediction
    return output, 201 