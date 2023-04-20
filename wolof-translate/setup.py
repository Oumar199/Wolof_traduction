from setuptools import setup

setup(name="wolof_translate", version="0.0.1", author="Oumar Kane", author_email="oumar.kane@univ-thies.sn", 
      description="Contain function and classes to process corpora for making translation between wolof text and other languages.",
      requires=['spacy', 'nltk', 'gensim'])
