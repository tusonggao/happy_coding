import time
import nltk
# import gensim
# import spacy
# import tensorflow as tf
import pandas as pd

print(nltk.__version__)

start_t = time.time()

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('treebank')
# nltk.download()

print('download cost time', time.time()-start_t)

sentence = """At eight o'clock on Thursday sunny morning 
     Arthur didn't feel very good. he said: I won't give up, never"""
tokens = nltk.word_tokenize(sentence)

print(tokens)

tagged = nltk.pos_tag(tokens)

print(tagged)

entities = nltk.chunk.ne_chunk(tagged)

print(entities)

# from nltk.corpus import treebank
# t = treebank.parsed_sents('wsj_0003.mrg')[0]
# t.draw()

print(nltk.corpus.gutenberg.fileids())