from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from data_loader import load_data
import numpy as np
import random

# Mean Square Error
def MSE(Y, Yhat):
    return np.mean((Y - Yhat)**2)

# Naive Predictor which outputs scores from the second period G2
def naive_predictor(Xmat):
    return Xmat["G2"]

# Tuning regularization hyperparameter for linear regression
def linear_regression(X_train, X_val, Y_train, Y_val, alphas):
    val_err = []
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, Y_train)
        val_err.append(MSE(Y_val, model.predict(X_val)))
    
    min_val_err = val_err.index(min(val_err))
    return alphas[min_val_err]

# Tuning tree depth for decision tree
def decision_tree(X_train, X_val, Y_train, Y_val):
    depths = [_ for _ in range(3, len(X_train.columns))]
    val_err = []

    for d in depths:
        model = DecisionTreeRegressor(max_depth=d, random_state=42)
        model.fit(X_train, Y_train)
        val_err.append(MSE(Y_val, model.predict(X_val)))
    
    min_val_err = val_err.index(min(val_err))
    return depths[min_val_err]

# Tuning tree depth for random forest
def random_forest(X_train, X_val, Y_train, Y_val):
    depths = [_ for _ in range(3, len(X_train.columns))]
    val_err = []

    for d in depths:
        model = RandomForestRegressor(n_estimators=500, max_depth=d, random_state=42)
        model.fit(X_train, Y_train)
        val_err.append(MSE(Y_val, model.predict(X_val)))

    min_val_err = val_err.index(min(val_err))
    return depths[min_val_err]

# Tuning the second hidden layer size for neural net with two hidden layers
def neural_net(X_train, X_val, Y_train, Y_val):
    second_layer_size = [2, 4, 8, 16, 32]
    val_err = []

    for size in second_layer_size:
        model = MLPRegressor(hidden_layer_sizes=(35,size), activation='relu', solver='adam', alpha=2, max_iter=10000, random_state=42)
        model.fit(X_train, Y_train)
        val_err.append(MSE(Y_val, model.predict(X_val)))
    
    min_val_err = val_err.index(min(val_err))
    return second_layer_size[min_val_err]

def main():
    random.seed(42)

    # Mathematics 
    print("Results for mathematics class: ")
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data("data/student-mat.csv")

    # Naive Predictor
    print("Naive predictor test err", MSE(Y_test, naive_predictor(X_test)))

    # Linear Regression
    best_alpha = linear_regression(X_train, X_val, Y_train, Y_val, [0, 0.05, 0.1, 0.5, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25])
    model = Ridge(alpha=best_alpha)
    model.fit(X_train, Y_train)
    print("Linear regression test err", MSE(Y_test, model.predict(X_test)))

    # Decision Tree
    best_depth = decision_tree(X_train, X_val, Y_train, Y_val)
    print("best_depth: ", best_depth)
    model = DecisionTreeRegressor(max_depth=best_depth)
    model.fit(X_train, Y_train)
    print("Decision tree test err", MSE(Y_test, model.predict(X_test)))

    # Random Forest
    best_depth = random_forest(X_train, X_val, Y_train, Y_val)
    print("best_depth: ", best_depth)
    model = RandomForestRegressor(max_depth=best_depth)
    model.fit(X_train, Y_train)
    print("Random forest test err", MSE(Y_test, model.predict(X_test)))

    # Neural Net
    best_second_layer_size = neural_net(X_train, X_val, Y_train, Y_val)
    print("best_second_layer_size: ", best_second_layer_size)
    model = MLPRegressor(hidden_layer_sizes=(35, best_second_layer_size), activation='relu', solver='adam', alpha=2, max_iter=10000, random_state=42)
    model.fit(X_train, Y_train)
    print("Neural net test err", MSE(Y_test, model.predict(X_test)))
    
    # Portugese
    print("\nResults for Portugese class: ")
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data("data/student-por.csv")

    # Naive Predictor
    print("Naive predictor test err", MSE(Y_test, naive_predictor(X_test)))

    # Linear Regression
    best_alpha = linear_regression(X_train, X_val, Y_train, Y_val, [0, 0.05, 0.1, 0.5, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25])
    model = Ridge(alpha=best_alpha)
    model.fit(X_train, Y_train)
    print("Linear regression test err", MSE(Y_test, model.predict(X_test)))
    
    # Decision Tree
    best_depth = decision_tree(X_train, X_val, Y_train, Y_val)
    print("best_depth: ", best_depth)
    model = DecisionTreeRegressor(max_depth=best_depth)
    model.fit(X_train, Y_train)
    print("Decision tree test err", MSE(Y_test, model.predict(X_test)))

    # Random Forest
    best_depth = random_forest(X_train, X_val, Y_train, Y_val)
    print("best_depth: ", best_depth)
    model = RandomForestRegressor(max_depth=best_depth)
    model.fit(X_train, Y_train)
    print("Random forest test err", MSE(Y_test, model.predict(X_test)))

    # Neural Net
    best_second_layer_size = neural_net(X_train, X_val, Y_train, Y_val)
    print("best_second_layer_size: ", best_second_layer_size)
    model = MLPRegressor(hidden_layer_sizes=(35, best_second_layer_size), activation='relu', solver='adam', alpha=2, max_iter=10000, random_state=42)
    model.fit(X_train, Y_train)
    print("Neural net test err", MSE(Y_test, model.predict(X_test)))

if __name__ == "__main__":
    main()