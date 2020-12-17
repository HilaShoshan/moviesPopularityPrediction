import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

class ArrangeData:

    df_x = 0
    df_y = 0

    def __init__(self, df, columns):
        df.dropna(0, subset=['release_date', 'runtime'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        self.df_x = df[columns]
        self.df_y = df[['popularity']]  # lable


    def split_date(self):
        """
            create two new columns to add the data set: by replacing release_date with: release_year and release_month
        """
        str_date = self.df_x[['release_date']].astype(str)
        years = []  # list that saves all the years from the date
        months = []  # list that saves all the months from the date
        for index in range(str_date.shape[0]):
            organ = str_date.iloc[:, 0].values[index]
            years.append(int(organ[:4]))
            months.append(int(organ[5:7]))
        self.df_x['release_year'] = years
        self.df_x['release_month'] = months
        del self.df_x['release_date']


    def label_encoding(self):
        for col in ['production_companies', 'genres', 'original_language', 'production_countries']:
            unique_vals = list(self.df_x[col].unique())
            dict = {}  # gives each unique value in column a unique number
            i = 0
            for x in unique_vals:
                dict[x] = int(i)
                i += 1
            self.df_x[col] = self.df_x[col].map(lambda x: dict.get(x))


    def one_hot_encoding(self):
        """
            using for original_language column
        """
        language_encoding = pd.get_dummies(self.df_x['original_language'], prefix='is')
        self.df_x = pd.concat([self.df_x, language_encoding], axis=1)
        del self.df_x['original_language']


    def encode_categorical_list_cols(self):
        new_df = pd.DataFrame(columns=[], index=range(self.df_x.shape[0]))  # create an empty DataFrame
        for col in ['production_companies', 'genres', 'production_countries']:  # the relevant columns
            index = -1
            for list in self.df_x[col]:  # its not really a list, but a string
                index += 1
                if len(list) == 2:  # empty list: []
                    continue
                organs = list[:len(list)-3].replace("[{", "").split('"name": ')  # list that contains different organs in different indices
                for organ in organs:
                    items = self.getItems(organ, col)
                    if len(items) == 0:  # items list is empty
                        continue
                    col_name = col[:len(col)-1] + str(items[0])
                    if col_name not in new_df.columns:  # we haven't created the column yet
                        new_df[col_name] = np.nan  # create it with nones
                    new_df.at[index, col_name] = 1
        new_df.fillna(0, inplace=True)
        self.df_x.drop(['production_companies', 'genres', 'production_countries'], axis=1, inplace=True)
        self.df_x = pd.concat([self.df_x, new_df], axis=1)


    def getItems(self, organ, col):
        """
            calls in encode_categorical_list_cols method
            :return: list of the names of companies/ genres/ countries
        """
        items = []
        if col == 'production_companies':
            items.append(organ.split('", "id":')[0][1:].replace(" ", ""))
        elif col == 'genres':
            if organ[:4] != '"id"':  # its not the first organ in organs
                items.append(organ.split('"}, {"id":')[0][1:].replace(" ", ""))
        else:  # col == 'production_countries'
            if organ[:4] == '"iso':
                items.append(organ.split('"iso_3166_1": "')[-1].replace('", ', ''))
        return items


    def fix_nulls(self):
        self.df_x['tagline'].fillna("", inplace=True)
        self.df_x['overview'].fillna("", inplace=True)


    def convert_with_nltk(self):
        """
            a method using to convert overview, title and tagline columns to numeric
        """
        pass


    def normalize(self):
        norm_df_x = (self.df_x - self.df_x.mean()) / self.df_x.std()
        return norm_df_x


    def arrange(self):
        self.split_date()
        self.fix_nulls()
        # print(self.df_x.isnull().sum())
        self.one_hot_encoding()
        self.encode_categorical_list_cols()
        self.convert_with_nltk()  # should replace the for loop
        for col in self.df_x.columns:
            if not is_numeric_dtype(self.df_x[col]):
                # print(col)
                del self.df_x[col]
        norm_df_x = self.normalize()
        return norm_df_x, self.df_y
