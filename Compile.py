from UoW.Connectors import Wrangler
from UoW.Rankings import SeasonRanks

if __name__ == "__main__":
    w = Wrangler()
    source = "raw"
    source_destination = "Data/RawData"
    # Step 2 (README)
    w.write_csv_to_folder(w.transform_br_data(w.many_seasons_to_1_file(source)), source_destination)

    rawdata = w.read_csv_from_folder(source_destination)
    o = SeasonRanks(rawdata)
    final_destination = "Data/FinalData"
    # Step 3 (README)
    #w.write_csv_to_folder(o.do_seasonal_ranking(), final_destination)
    o.do_2_seasonal_ranking()
