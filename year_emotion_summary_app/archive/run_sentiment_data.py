import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline , pipeline
from merge_data import join_data
import time



summarizer = pipeline("summarization",model="slauw87/bart_summarisation", max_length=50, truncation=True)
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=6, max_length=10, truncation=True)

df = join_data('data/t*.csv')

def summarize_text(text):
    return summarizer(text)[0]['summary_text']

def get_sentiments(text):
    predictions = pipe(text)[0]
    result_dict = {i['label']:i['score'] for i in predictions}
    verdict = predictions[0]['label']
    neg_score = result_dict['negative']
    pos_score = result_dict['positive']
    neu_score = result_dict['neutral']
    return verdict, neg_score, pos_score, neu_score


def extract_sentiments(text):
    try:
        answer = get_sentiments(text)
    except:
        summary = summarize_text(text)
        answer = get_sentiments(summary) 
    return answer

def run_function():
    list_dump = []
    list_of_text = list(df['text'])
    count = 0
    end = 1000
    start_program = time.time()
    for i in range(50):
        # summarizer = pipeline("summarization",model="slauw87/bart_summarisation")
        # tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        # model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        # pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=20)
        start_time = time.time()
        new_list = []
        new_list = list_of_text[count:end]
        print(f"splitted {i}")
        for text in new_list:
            list_dump.append(extract_sentiments(text)) 
            
        # print(f'done sentiment with {i} list')
        
        count = end
        end += 1000
        end_time = time.time()
        print(f"completed {i} at {str(end_time-start_time)}")
        print("-------------------------")
    end_program = time.time()
    print(f"all completed at {str(end_program-start_program)}")
    df[["verdict", "neg_score", 'pos_score', 'neu_score']] = list_dump
    df.to_csv('data/all_data.csv', index=False)
    print("completed saving data")


if __name__== "__main__":
    run_function()

    # list_dump = []
    # list_of_text = list(df['text'])
    # count = 0
    # end = 100
    # start_program = time.time()
    # for i in range(500):
    #     summarizer = pipeline("summarization",model="slauw87/bart_summarisation")
    #     tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    #     model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    #     pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=20)
    #     start_time = time.time()
    #     new_list = []
    #     new_list = list_of_text[count:end]
    #     print(f"splitted {i}")
    #     for text in new_list:
    #         list_dump.append(extract_sentiments(text)) 
            
    #     # print(f'done sentiment with {i} list')
        
    #     count = end
    #     end += 100
    #     end_time = time.time()
    #     print(f"completed {i} at {str(end_time-start_time)}")
    #     print("-------------------------")
    # end_program = time.time()
    # print(f"all completed at {str(end_program-start_program)}")
    # df[["verdict", "neg_score", 'pos_score', 'neu_score']] = list_dump
    # df.to_csv('data/all_data.csv', index=False)
    # print("completed saving data")