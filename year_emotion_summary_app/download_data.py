"""
To start the environment by installing the libraries and downloading the data needed

"""

import subprocess

CMD = """
pip install -r requirements.txt;
mkdir .kaggle;
cp ../kaggle.json ~/.kaggle/kaggle.json;
kaggle datasets download -d thedevastator/imdb-large-movie-review-dataset-binary-sentiment;
unzip imdb-large-movie-review-dataset-binary-sentiment.zip;
mkdir data
mv imdb-large-movie-review-dataset-binary-sentiment.zip *.csv data/
"""

if __name__== "__main__":
    ret = subprocess.run(CMD, capture_output=True, shell=True)
    
