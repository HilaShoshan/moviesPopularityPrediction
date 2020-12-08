import pandas as pd
from pandas.api.types import is_numeric_dtype

class ArrangeData:

    df_x = 0
    df_y = 0

    def __init__(self, df, columns):
        df = df.dropna(0, subset=['release_date', 'runtime'])
        self.df_x = df[columns]
        self.df_y = df[['popularity']]  # lable

    """
        function that create two new columns to add the data set, 
        replace release_date with: release_year and release_month
    """
    def split_date(self):
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


    def convert_to_numeric(self):  # need to convert it to isEnglish, isFranch ect.
        for col in ['production_companies', 'genres', 'original_language', 'production_countries']:
            unique_vals = list(self.df_x[col].unique())
            dict = {}  # gives each unique value in column a unique number
            i = 0
            for x in unique_vals:
                dict[x] = int(i)
                i += 1
            self.df_x[col] = self.df_x[col].map(lambda x: dict.get(x))


    def fix_nulls(self):
        self.df_x['tagline'].fillna("", inplace=True)
        self.df_x['overview'].fillna("", inplace=True)


    def normalize(self):
        norm_df_x = (self.df_x - self.df_x.mean()) / self.df_x.std()
        norm_df_y = (self.df_y - self.df_y.mean()) / self.df_y.std()
        return norm_df_x, norm_df_y


    def convert_with_nltk(self):
        pass


    def arrange(self):
        self.split_date()
        self.convert_to_numeric()  # We will improve it later so that the values will be numerically better
        self.fix_nulls()
        # print(self.df_x.isnull().sum())
        """
        for col in self.df_x.columns:
            if not is_numeric_dtype(self.df_x[col]):
                # print(col)
                del self.df_x[col]
        """
        norm_df_x, norm_df_y = self.normalize()
        return norm_df_x, norm_df_y