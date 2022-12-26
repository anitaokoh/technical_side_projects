"""
To control the frontend part of the application using streamlit and connect all the other modules 
"""
import streamlit as st
import pandas as pd
import numpy as np
from sentiment_model import extract_sentiments
from predict_emotion import model_wrap
from plotly_viz import create_fig
from  download_viz import get_table_download_link
import json

summarizer = pipeline("summarization",model="slauw87/bart_summarisation", max_length=50, truncation=True)
tokenizer = AutoTokenizer.from_pretrained('philschmid/distilbert-base-multilingual-cased-sentiment-2')
model = AutoModelForSequenceClassification.from_pretrained('philschmid/distilbert-base-multilingual-cased-sentiment-2')
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=6, max_length=10, truncation=True)


with open('model/25-12-2022-08:30:05_cluster_map.json', 'r') as openfile:
    # Reading from json file
    emotion_map = json.load(openfile)

#App data
emotion_index_map = {"esctastic/joyful": 5, "happy/content": 4,"Ok/Meh": 3,"sad/anxious": 2,"depressed/frustrated": 1}
emotions_list = list(emotion_index_map.keys())
emotion_scale = list(emotion_index_map.values())
month_list = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']
month_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]



@st.cache
def run_emotion_sent(data:list)->list:
    """
    To get the sentiment score of the user text input list , predict each of the text clusters
    and map their emotions
    """
    emotion_pred = []
    for text in data:
        neg_score, pos_score = extract_sentiments(text)
        result = model_wrap([neg_score, pos_score], emotion_map)
        emotion_pred.append(result)
    return emotion_pred

@st.cache
def viz_page():
    """
    Get a downloadable Chart image of the months' ratings for the year by calling the above functions.
    Args: 
    Return: A download link to the wrap up chart
    """

    #the first part
    max_width_str = f"max-width: 1030px;"
    st.markdown(f"""<style>.reportview-container .main .block-container{{{max_width_str}}}</style>""",unsafe_allow_html=True)
    st.image('images/moritz-knoringer-4_MwbIq0CME-unsplash.jpg',use_column_width=True)
    st.subheader('How has your 2022 been?')
    st.markdown("""<p> 2022 has been a year filled with rollercoasts . Some good and some sad emotions</p> 
        <p>This web app helps predict the sentiment of your monthly summary and maps/visualizes each sentiment into  five emotions </p>
        <p> This emotions are : <strong>Esctastic/Joyful,Happy/Content', Ok/Meh, Sad/Anxious</strong> and <strong>Depressed/Frustrated</strong> </p>
        <p> Wanna give it a try? Go ahead and tick the start checkbox below</p>
         """,unsafe_allow_html=True)

    # the second part for the ratings
    if st.checkbox('Start'):
        st.write("""
        Take a moment to write briefly a summary of each month . It could be a positive event that summarizes your month like "In january , 
        I got a new job offer that made my whole month" or a negative event like " in March, i spent my time living on the streets because i was homeless".
        You also do not need to fill up all the month if you don't remember. You can leave the default message " Don't remember"
        """)
        
        col1, col2 = st.columns(2)
        data_input1 = []
        data_input2 =[]
        col1.subheader('First Half of the Year')
        col2.subheader('Second Half of the Year')
        for element in month_list[:6]:
            element = col1.text_area(element, "Don't remember")
            data_input1.append(element)  
        for element in month_list[6:]:
            element = col2.text_area(element, "Don't remember ")
            data_input2.append(element)
        data = data_input1+data_input2 
        emotion_pred = []
        if st.checkbox('Next'):
            emotion_pred = run_emotion_sent(data)
            final_df = dict(zip(month_list, emotion_pred))
            display_df = pd.DataFrame(list(final_df.items()),columns=['Month', 'Emotion'])
            display_df['Month_Index'] = month_index
            display_df['Emotion_Scale'] = display_df['Emotion'].map(emotion_index_map)
            display_df['summary'] = data
            with st.expander('See Chart'):  

                    fig = create_fig(display_df, emotions_list,emotion_scale)
                    st.plotly_chart(fig,use_container_width=True,config={'displayModeBar': False})
                    if st.button('Export Vizualization image'):
                        st.text('Ready to Download')
                        st.markdown(get_table_download_link(fig), unsafe_allow_html=True)



if __name__ == "__main__":
    viz_page()
