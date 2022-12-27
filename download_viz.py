"""
To download the visualization created

"""

import plotly.io
import base64


def get_table_download_link(data):
    """
    Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
#     figure = data.to_image(format="png",  engine="kaleido")
    figure = plotly.io.to_image(data,format="png",  engine="kaleido")
#     figure =  py.get(data)
    b64 = base64.b64encode(figure).decode()  
    href = f'<a href="data:file/image;base64,{b64}" download="My 2022 in a wrap.png">Download chart</a>'
    return href