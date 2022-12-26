"""
To predict the sentiment to a cluster and map the emotion
"""

import sys
from mlem.api import apply
import json

with open('model/25-12-2022-08:30:05_cluster_map.json', 'r') as openfile:
    # Reading from json file
    emotion_map = json.load(openfile)


def model_wrap(arr:list, map_key:dict)->str:
    """
    Predict the sentiment score into a cluster and map the emotion
    """
    prediction= apply('model/25-12-2022-09:20:44_kmeans_model',[arr],method='predict')
    return map_key[str(prediction[0])]


if __name__== "__main__":
   pred = model_wrap([float(sys.argv[1]),float(sys.argv[2])] , emotion_map)
   print(pred)



