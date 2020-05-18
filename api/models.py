from . import db
from sqlalchemy.types import JSON

class Article(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    accuracy = db.Column(db.Integer)