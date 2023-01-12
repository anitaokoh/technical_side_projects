# Year Emotion Summarization App
![image](images/moritz-knoringer-4_MwbIq0CME-unsplash.jpg)
This app (powered by streamlit) helps predict the sentiment of your monthly summary and maps each sentiment into five emotions which are 
- Estatic/Joyful
- Happy/ Content
- Ok/Meh
- Sad/Anxious
- Depressed/ Frustrated

All this emotions and month are they visualized into a line chart as a summary for the year

You can find the link to the app [here](http://52.90.71.18:8501/)

### Video of the app
![streamlitapp](/streamlitapp.gif)

Find article documentation [here](https://medium.com/data-de-mystified/building-a-text-emotion-analyser-app-c4858c790fae)

**Note that the link may go offline depending on when in the future you read this. If it is unavailable , feel free to clone this repository and run the `main_app.py` file.(Be sure to pip install all the libraries in the `requirement.txt` file before hand)**

### Flow of Project
The project can be split into 2 parts chronologically.
- Model building and emotion mapping
- Front end app and Visualization of emotion scale

### Model Building and emotion mapping
- The enviroment and data download is done using the `download.py` file. All data used are saved in the folder called `data`. All libraries used are stored in `requirement.txt`file
- In order to get more training data, the `training.csv` and `test.csv` in the image folder was merged into one file  using the `merge_data.py`. This result table has 50000 rows and two columns( text and label) consisting of 50% each positive and negative ground labels. Source data can be found in kaggle [here](https://www.kaggle.com/datasets/thedevastator/imdb-large-movie-review-dataset-binary-sentiment)
- The model used for the text sentiment is TextBlob library. Using the `text_blob_sent.py` , all rows were ran resulting to producing the negative and positive scores of each text
- The evaluation of the textblob sentiments and the building/saving of the kmeans model was captured in the `building_model_and_evaluation.ipynb` notebook
- All model and emotion mapping files are saved in the folder called `model`

### Front end app and Visualization of emotion scale
- The `main_app.py` file creates the streamlit app and controls all tasks in the app
- The `sentiment_model.py` and the `predict_emotion.py` are used to predict emotions of text written on the app
- The `plotly_viz.py` creates the visualization and the `download_viz.py` file creates a downloadable link for the app.
- All images are stored in the image folder

One thing to note
- For the Model building part, the sentiment model used was Textblob . However for the realtime / front-end app part, the model used is Hugging face model called `philschmid/distilbert-base-multilingual-cased-sentiment-2`. This is because textblob was not performing so well as sentiments that were clearly negative were labelled as positive. Switching the models did not seem to after the kmeans clustering predictions

