from UoW.Connectors import Wrangler
from UoW.Rankings import OneSeason

if __name__ == "__main__":
    # For NBA br data: compile many seasons from raw folder into 1 csv
    # Then put 1 file named Raw Data in Data folder
    w = Wrangler()
    source = "raw"
    source_destination = "Data/Raw Data"
    final_destination = "Data/Final Data"
    w.write_csv_to_folder(
        w.transform_br_data(
            w.many_seasons_to_1_file(source)
        )
        ,
        source_destination
    )

    # Generating features
    rawdata = w.read_csv_from_folder(source_destination)
    o = OneSeason(rawdata)

    w.write_csv_to_folder(
            o.do_seasonal_ranking(),
            final_destination
    )