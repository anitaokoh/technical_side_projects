"""
To get the sentiments of the user text in real time
"""

import pandas as pd
import numpy as np
from textblob import Blobber, TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import time
import nltk
from transformers import  pipeline
nltk.download('movie_reviews')
nltk.download('punkt')


summarizer = pipeline("summarization",model="slauw87/bart_summarisation", max_length=50, truncation=True)

tb = Blobber(analyzer=NaiveBayesAnalyzer())

def get_sentiments(text:str)->tuple:
    """
    To get the sentiment scores of the text . If the text is longer than 400 characters, it is first summarized and
    then predicted for the sentiment scores
    
    """
    if len(text)>=400:
        summary = summarizer(text)[0]['summary_text']
    else:
        summary = text
    score = tb(summary).sentiment
    pos_score = score[1]
    neg_score = score[2]
    return  neg_score, pos_score

if __name__== "__main__":  
    neg_score, pos_score = get_sentiments("June was quite a depressing month I was sick the entire month it was so frustrating")
    print(neg_score, pos_score)

