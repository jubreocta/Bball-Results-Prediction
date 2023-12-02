from UoW.Connectors import Wrangler

#For NBA br data: compile many seasons from raw folder into 1 csv
#then put 1 file named Raw Data in Data folder
W = Wrangler()
W.write_csv_to_folder(W.many_seasons_to_1_file("raw"), "Data/Raw Data")