import pandas as pd
import dateutil.parser


def parse_yt_datetime(date):
    return dateutil.parser.parse(date)

def parse_and_concat_csvs(list_filenames, header):
    dataframes = []
    for filename in list_filenames:
        dataframes.append(pd.read_csv(filename, names=header))
    return pd.concat(dataframes)


