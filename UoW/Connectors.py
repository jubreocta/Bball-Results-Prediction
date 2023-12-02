import os
import pandas as pd

class Wrangler:
    def __init__(self):
        pass
    def many_seasons_to_1_file(self, folder):
        return pd.concat([pd.read_csv(f"{folder}/{f}").assign(Season=f[:-4]) for f in os.listdir(folder)], ignore_index=False)
    
    def write_csv_to_folder(self, file, rel_path):
        file.to_csv(f"{rel_path}.csv")