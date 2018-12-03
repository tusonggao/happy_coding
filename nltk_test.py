import nltk
# import gensim
# import spacy
# import tensorflow as tf
import pandas as pd

print(nltk.__version__)

sentence = """At eight o'clock on Thursday morning 
     Arthur didn't feel very good."""
tokens = nltk.word_tokenize(sentence)

print(tokens)