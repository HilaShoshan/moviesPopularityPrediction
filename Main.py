from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics

from LinearRegression import *
from ArrangeData import *
from MultiLayerPerceptron import *


df = pd.read_csv("tmdb_5000_movies.csv")
columns = ['runtime', 'production_companies', 'genres', 'revenue', 'original_language', 'overview',
         'production_countries', 'release_date', 'vote_count', 'vote_average', 'title', 'tagline', 'budget']


def compute_error(y_test, y_pred):
    #print(metrics.accuracy_score(y_test, y_pred))
    #print(metrics.confusion_matrix(y_test, y_pred))
    print(mean_squared_error(y_test, y_pred))


def main():
    data = ArrangeData(df, columns)
    norm_df_x, norm_df_y = data.arrange()

    # split data to training set, testing set and validation set
    X_train, X_test, y_train, y_test = train_test_split(norm_df_x, norm_df_y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    # Linear Regression Model
    """
    print("Linear Regression Model")
    W, b = linearRegression(X_train, y_train)
    y_pred = predict_linear(W, b, X_test)
    compute_error(y_test, y_pred)
    print("___________________________________")
    print(y_test)
    print("___________________________________")
    print(y_pred)
    

    # MLP Model
    print("MLP Model")
    W1, b1, W2, b2 = MLP(X_train, y_train)
    y_pred = predict_MLP(W1, b1, W2, b2, X_test)
    compute_error(y_test, y_pred)
"""

if __name__ == '__main__':
    main()

