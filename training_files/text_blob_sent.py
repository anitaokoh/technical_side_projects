"""
To get the sentiments of the training and test data which would be further used to create the cluster model
"""

import pandas as pd
import numpy as np
from textblob import Blobber, TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from merge_data import join_data
import time
import nltk
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
nltk.download('movie_reviews')
nltk.download('punkt')

df = join_data('data/t*.csv')


tb = Blobber(analyzer=NaiveBayesAnalyzer())
def get_sentiments(text:str)->tuple:
    """
    to get the text sentiment score and label
    """
    score = tb(text).sentiment
    verdict = score[0]
    pos_score = score[1]
    neg_score = score[2]
    return verdict, pos_score, neg_score


if __name__== "__main__":   
    start_time = time.time()
    df[["verdict", 'pos_score', 'neg_score']] = df.parallel_apply(lambda x: get_sentiments(x.text), axis=1, result_type="expand")
    end_time = time.time()
    print(f"completed at {str(end_time-start_time)}")
    df.to_csv('data/all_data_textblob.csv', index=False)
    print("completed saving data")

