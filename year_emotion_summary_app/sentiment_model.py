import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline , pipeline

# summarizer = pipeline("summarization",model="slauw87/bart_summarisation", max_length=50, truncation=True)
# tokenizer = AutoTokenizer.from_pretrained('philschmid/distilbert-base-multilingual-cased-sentiment-2')
# model = AutoModelForSequenceClassification.from_pretrained('philschmid/distilbert-base-multilingual-cased-sentiment-2')
# pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=6, max_length=10, truncation=True)



def summarize_text(text):
    return summarizer(text)[0]['summary_text']

def get_sentiments(text):
    predictions = pipe(text)[0]
    result_dict = {i['label']:i['score'] for i in predictions}
    # verdict = predictions[0]['label']
    neg_score = result_dict['negative']
    pos_score = result_dict['positive']
    # neu_score = result_dict['neutral']
    return neg_score, pos_score


def extract_sentiments(text):
    try:
        answer = get_sentiments(text)
    except:
        summary = summarize_text(text)
        answer = get_sentiments(summary) 
    return answer
