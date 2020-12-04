import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


df = pd.read_csv("tmdb_5000_movies.csv")
columns = ['runtime', 'production_companies', 'genres', 'revenue', 'original_language', 'overview',
         'production_countries', 'release_date', 'vote_count', 'vote_average', 'title', 'tagline', 'budget']
df_x = df[columns]
df_y = df[['popularity']]  # lable
df_x = df_x.dropna(0, subset=['release_date'])


"""
    function that create two new columns to add the data set, 
    replace release_date with: release_year and release_month
"""
def split_date():
    str_date = df_x[['release_date']].astype(str)
    years = []  # list that saves all the years from the date
    months = []  # list that saves all the months from the date
    for index in range(str_date.shape[0]):
        organ = str_date.iloc[:, 0].values[index]
        years.append(int(organ[:4]))
        months.append(int(organ[5:7]))
    df_x['release_year'] = years
    df_x['release_month'] = months
    del df_x['release_date']


def convert_to_numeric():  # need to convert it to isEnglish, isFranch ect.
    for col in ['production_companies', 'genres', 'original_language', 'production_countries']:
        unique_vals = list(df[col].unique())
        dict = {}  # gives each unique value in column a unique number
        i = 0
        for x in unique_vals:
            dict[x] = int(i)
            i += 1
        df_x[col] = df_x[col].map(lambda x: dict.get(x))


def main():
    split_date()
    convert_to_numeric()  # We will improve it later so that the values will be numerically better
    for col in df_x.columns:
        if not is_numeric_dtype(df_x[col]):
            print(col)


if __name__ == '__main__':
    main()

