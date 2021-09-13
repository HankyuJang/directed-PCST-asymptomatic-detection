import pandas as pd

# filter in records in df from start_date until the day before the end date
def filter_records(df, start_date, end_date):
    df_temp = df[(start_date <= df["date"]) & (df["date"] < end_date)]
    df_temp.reset_index(drop=True, inplace=True)
    return df_temp

# Concatenate two dataframes. 
# Two dataframes must have same column names
def concat_dataframes(df1, df2):
    df_temp = pd.concat([df1, df2])
    df_temp.reset_index(drop=True, inplace=True)
    return df_temp
