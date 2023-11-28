import os
import re
import pandas as pd

rank_votings = dict()
raw_folder = "raw"
raw_directory_list = os.listdir(raw_folder)
for file in raw_directory_list:    
    data = pd.read_csv(f"{raw_folder}/{file}")
    print(data.head())