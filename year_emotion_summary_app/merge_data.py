"""
To merge the training and test data together for textblob sentiment prediction

"""
import os
import glob
import pandas as pd




def get_file_list(table_path:str)->list:
    """
    Get all the data files and store in a list
    """
    return glob.glob(f'{table_path}')

def join_data(table_path:str):
    """
    Merge all the focus data files into one dataframe
    """
    df_append = pd.DataFrame()
    for file in get_file_list(table_path):
                df_temp = pd.read_csv(file)
                df_append = df_append.append(df_temp, ignore_index=True)
    return df_append 

if __name__== "__main__":
#append all files together
    df = join_data('data/t*.csv')
    
