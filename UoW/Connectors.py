import os
import pandas as pd
import datetime
class Wrangler:
    def __init__(self):
        pass
    def many_seasons_to_1_file(self, folder):
        return pd.concat([pd.read_csv(f"{folder}/{f}").assign(Season=f[:-4]) for f in os.listdir(folder)], ignore_index=False)

    def read_csv_from_folder(self, rel_path):
        return pd.read_csv(f"{rel_path}.csv")

    def write_csv_to_folder(self, file, rel_path):
        file.to_csv(f"{rel_path}.csv", index = False)

    def replacement_dictionary(self):
        return {
            "New Jersey Nets":"Brooklyn Nets",
            "New Orleans Hornets": "New Orleans Pelicans",
            "Seattle SuperSonics":"Oklahoma City Thunder",
            "New Orleans/Oklahoma City Hornets":"New Orleans Pelicans",
            "Charlotte Bobcats": "Charlotte Hornets"
        }

    def transform_br_data(self, data):
        data["Date"] = data["Date"]+ " " + data["Start (ET)"] + "M"
        data["Date"] = data["Date"].map(lambda x: datetime.datetime.strptime(x, "%a %b %d %Y %H:%M%p").strftime("%d/%m/%Y %H:%M"))
        data["overtime"] = data["Unnamed: 7"]
        data[["Date","Time"]] = data["Date"].str.split(" ",expand=True,)
        try:
            data[["LeftScore","RightScore"]] = data.Result.str.split(" - ",expand=True,)
        except:
            data[["LeftScore","RightScore"]] = (None,None)
        data["LeftTeam"]   = data["Home/Neutral"].replace(self.replacement_dictionary())
        data["RightTeam"]  = data["Visitor/Neutral"].replace(self.replacement_dictionary())
        data["Date"]       = pd.to_datetime(data["Date"], format = "%d/%m/%Y")
        data["LeftScore"]  = pd.to_numeric(data["PTS.1"])
        data["RightScore"] = pd.to_numeric(data["PTS"])
        data.reset_index(inplace = True, drop = True)
        data = data[[
            "Season",
            "Date",
            "LeftTeam",
            "LeftScore",
            "RightScore",
            "RightTeam"
        ]]
        return data