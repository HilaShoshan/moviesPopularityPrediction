from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from LinearRegression import *
from ArrangeData import *
from MultiLayerPerceptron import *
from NeuralNetwork import *

from sklearn.linear_model import LinearRegression


df = pd.read_csv("tmdb_5000_movies.csv")
columns = ['runtime', 'production_companies', 'genres', 'revenue', 'original_language', 'overview',
         'production_countries', 'release_date', 'vote_count', 'vote_average', 'title', 'tagline', 'budget']


def compute_error(y_real, y_pred):  # The mean squared error
    print("MSE:", mean_squared_error(y_real, y_pred))
    print("MAE:", mean_absolute_error(y_real, y_pred))


def plot_err(epochs, train_err, test_err):
    plt.xlabel('epochs')
    plt.ylabel('prediction error')
    plt.title('Train/Test error')
    plt.plot(epochs, train_err, color='blue', linewidth = 3,  label = 'train error')
    plt.plot(epochs, test_err, color='red', linewidth = 3,  label = 'test error')
    plt.legend()
    plt.show()


def avg_baseline_pred(y_train, shape):
    train_avg = y_train.mean(axis=0)
    avg_pred = np.full(shape, train_avg)
    return avg_pred


def main():
    data = ArrangeData(df, columns)
    norm_df_x, df_y = data.arrange()

    # split data to training set, testing set and validation set

    X_train, X_test, y_train, y_test = train_test_split(norm_df_x, df_y, test_size=0.2, random_state=1)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    # Linear Regression Model
    print("Linear Regression Model")
    W, b, epochs, train_err, test_err = train_linreg(X_train, y_train, X_test, y_test, None, "adam")
    y_pred = predict_linreg(W, b, X_test)
    compute_error(y_test, y_pred)
    plot_err(epochs, train_err, test_err)

    """
    # Compare to the average baseline:
    print("Average Baseline")
    avg_pred = avg_baseline_pred(y_train, y_test.shape)
    compute_error(y_test, avg_pred)

    # MLP Model
    print("MLP Model")
    W1, b1, W2, b2, epochs, train_err, test_err = train_MLP(X_train, y_train, X_test, y_test)
    y_pred = predict_MLP(W1, b1, W2, b2, X_test)
    compute_error(y_test, y_pred)
    plot_err(epochs, train_err, test_err)

    
    # NN Model
    print("NN Model")
    W1, b1, W2, b2, W3, b3, epochs, train_err, test_err = train_NN(X_train, y_train, X_test, y_test, None, "adam")
    y_pred = predict_NN(W1, b1, W2, b2, W3, b3, X_test)
    compute_error(y_test, y_pred)
    plot_err(epochs, train_err, test_err)
"""

if __name__ == '__main__':
    main()

