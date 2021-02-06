from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ArrangeData import *
from LinearRegression import *
from NeuralNetwork import *
from RNN import *

import chazutsu
r = chazutsu.datasets.MovieReview.polarity().download()

df = pd.read_csv("tmdb_5000_movies.csv")
columns = ['runtime', 'production_companies', 'genres', 'original_language', 'production_countries',
           'release_date', 'vote_count', 'vote_average', 'budget']


def compute_error(y_real, y_pred):
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    print("MSE:", mse)
    print("MAE:", mean_absolute_error(y_real, y_pred))
    print("rMSE:", rmse)
    print("R^2 score:", 1-rmse)


def plot_err(epochs, train_err, test_err, model_name):
    plt.xlabel('epochs')
    plt.ylabel('prediction error')
    plt.title('Train/Test error '+model_name)
    plt.plot(epochs, train_err, color='blue', linewidth=3,  label='train error')
    plt.plot(epochs, test_err, color='red', linewidth=3,  label='test error')
    plt.legend()
    plt.show()


def avg_baseline_pred(y_train, shape):
    train_avg = y_train.mean(axis=0)
    avg_pred = np.full(shape, train_avg)
    return avg_pred


def fix_skew(y_train, y_test, show=False):  # not in use
    y_train = np.log(y_train)
    y_test = np.log(y_test)
    if show:
        y_train.plot(kind='hist', figsize=(8, 8))
        plt.xlabel('Popularity')
        plt.ylabel('# of movies')
        plt.show()
    return y_train, y_test


def main():
    """
    data = ArrangeData(df, columns)
    norm_df_x, df_y = data.arrange()

    # split data to training set, testing set and validation set

    X_train, X_test, y_train, y_test = train_test_split(norm_df_x, df_y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2

    # y_train, y_test = fix_skew(y_train, y_test)
    
    # Linear Regression Model
    print("Linear Regression Model")
    W, b, epochs, train_err, test_err = train_linreg(X_train, y_train, X_test, y_test, regularization="lasso")
    y_pred = predict_linreg(W, b, X_test)
    compute_error(y_test, y_pred)
    plot_err(epochs, train_err, test_err, "Linear Regression")

    # Compare to the average baseline:
    print("Average Baseline")
    avg_pred = avg_baseline_pred(y_train, y_test.shape)
    compute_error(y_test, avg_pred)
    
    # NN Model
    print("NN Model")
    Ws, biases, epochs, train_err, test_err = train_NN(X_train, y_train, X_test, y_test, num_epochs=100)
    y_pred = predict_NN(Ws, biases, X_test)
    compute_error(y_test, y_pred)
    plot_err(epochs, train_err, test_err, "NN with 2 hidden layers")

    # with early stopping
    Ws, biases, epochs, train_err, test_err = train_NN(X_train, y_train, X_val, y_val, num_epochs=100, early_stopping=True)
    y_pred = predict_NN(Ws, biases, X_test)
    compute_error(y_test, y_pred)
    plot_err(epochs, train_err, test_err, "NN with 2 hidden layers and early stopping")
"""
    print("RNN on overview column")
    train_RNN(df)


if __name__ == '__main__':
    main()

